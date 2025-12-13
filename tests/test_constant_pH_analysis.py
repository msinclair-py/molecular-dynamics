"""
Unit tests for analysis/constant_pH_analysis.py module
"""
import pytest
import numpy as np
import polars as pl
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import warnings


class TestUWHAMSolver:
    """Test suite for UWHAMSolver class"""
    
    def test_uwham_solver_init(self):
        """Test UWHAMSolver initialization"""
        from molecular_simulations.analysis.constant_pH_analysis import UWHAMSolver
        
        solver = UWHAMSolver(tol=1e-6, maxiter=5000)
        
        assert solver.tol == 1e-6
        assert solver.maxiter == 5000
        assert solver.f is None
        assert np.isclose(solver.log10, np.log(10))
    
    def test_uwham_solver_load_data(self):
        """Test UWHAMSolver load_data method"""
        from molecular_simulations.analysis.constant_pH_analysis import UWHAMSolver
        
        solver = UWHAMSolver()
        
        # Create test DataFrame
        data = {
            'rankid': [0, 0, 0, 1, 1, 1],
            'current_pH': [4.0, 4.0, 4.0, 7.0, 7.0, 7.0],
            'res1': [1, 0, 1, 0, 0, 1],
            'res2': [1, 1, 0, 0, 1, 0],
        }
        df = pl.DataFrame(data)
        
        solver.load_data(df, ['res1', 'res2'])
        
        assert len(solver.pH_values) == 2
        assert 4.0 in solver.pH_values
        assert 7.0 in solver.pH_values
        assert solver.nstates == 2
    
    def test_uwham_solver_solve(self):
        """Test UWHAMSolver solve method"""
        from molecular_simulations.analysis.constant_pH_analysis import UWHAMSolver
        
        solver = UWHAMSolver(tol=1e-5, maxiter=100)
        
        # Create simple test data
        data = {
            'rankid': [0] * 10 + [1] * 10,
            'current_pH': [4.0] * 10 + [7.0] * 10,
            'res1': [1] * 5 + [0] * 5 + [0] * 5 + [1] * 5,
        }
        df = pl.DataFrame(data)
        
        solver.load_data(df, ['res1'])
        
        # Should run without error
        f = solver.solve(verbose=False)
        
        assert f is not None
        assert len(f) == solver.nstates
        assert f[0] == 0  # Normalized so f[0] = 0
    
    def test_uwham_solver_compute_log_weights_before_solve(self):
        """Test that compute_log_weights raises error before solve"""
        from molecular_simulations.analysis.constant_pH_analysis import UWHAMSolver
        
        solver = UWHAMSolver()
        
        with pytest.raises(RuntimeError, match="Must call solve"):
            solver.compute_log_weights(5.0)
    
    def test_uwham_solver_compute_log_weights(self):
        """Test compute_log_weights after solving"""
        from molecular_simulations.analysis.constant_pH_analysis import UWHAMSolver
        
        solver = UWHAMSolver(tol=1e-5, maxiter=100)
        
        # Create test data
        data = {
            'rankid': [0] * 10 + [1] * 10,
            'current_pH': [4.0] * 10 + [7.0] * 10,
            'res1': [1] * 5 + [0] * 5 + [0] * 5 + [1] * 5,
        }
        df = pl.DataFrame(data)
        
        solver.load_data(df, ['res1'])
        solver.solve(verbose=False)
        
        log_weights, log_norm = solver.compute_log_weights(5.5)
        
        assert len(log_weights) == 20  # Total samples
        assert isinstance(log_norm, float)
    
    def test_uwham_solver_compute_expectation(self):
        """Test compute_expectation_at_pH method"""
        from molecular_simulations.analysis.constant_pH_analysis import UWHAMSolver
        
        solver = UWHAMSolver(tol=1e-5, maxiter=100)
        
        data = {
            'rankid': [0] * 10 + [1] * 10,
            'current_pH': [4.0] * 10 + [7.0] * 10,
            'res1': [1] * 5 + [0] * 5 + [0] * 5 + [1] * 5,
        }
        df = pl.DataFrame(data)
        
        solver.load_data(df, ['res1'])
        solver.solve(verbose=False)
        
        # Observable: the residue state itself
        observable = [solver.states['res1'][0], solver.states['res1'][1]]
        
        expectation = solver.compute_expectation_at_pH(observable, 5.5)
        
        assert isinstance(expectation, float)
        assert 0 <= expectation <= 1  # Should be a probability
    
    def test_uwham_solver_get_occupancy(self):
        """Test get_occupancy_for_resid method"""
        from molecular_simulations.analysis.constant_pH_analysis import UWHAMSolver
        
        solver = UWHAMSolver()
        
        data = {
            'rankid': [0] * 5 + [1] * 5,
            'current_pH': [4.0] * 5 + [7.0] * 5,
            'res1': [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        }
        df = pl.DataFrame(data)
        
        solver.load_data(df, ['res1'])
        
        occupancy = solver.get_occupancy_for_resid('res1')
        
        assert len(occupancy) == 2  # Two pH values
        assert len(occupancy[0]) == 5
        assert len(occupancy[1]) == 5


class TestTitrationCurve:
    """Test suite for TitrationCurve class"""
    
    def create_test_log_file(self, tmpdir, n_pH=5, n_samples=10):
        """Helper to create a test log file"""
        log_path = Path(tmpdir) / 'cpH.log'
        
        lines = [
            "cpH: resids 20  76  83\n"
        ]
        
        pH_values = np.linspace(2.0, 10.0, n_pH)
        for pH in pH_values:
            for i in range(n_samples):
                # Generate random states (0 or 1)
                states = [np.random.randint(0, 2) for _ in range(3)]
                lines.append(f"rank=0 cpH: pH {pH:.1f}: {states}\n")
        
        log_path.write_text(''.join(lines))
        return log_path
    
    def test_titration_curve_parse_log(self):
        """Test parsing of constant pH log file"""
        from molecular_simulations.analysis.constant_pH_analysis import TitrationCurve
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple log file
            log_path = Path(tmpdir) / 'cpH.log'
            log_content = """cpH: resids 20  76  83
rank=0 cpH: pH 4.0: [1, 1, 0]
rank=0 cpH: pH 4.0: [1, 0, 1]
rank=0 cpH: pH 7.0: [0, 0, 1]
rank=0 cpH: pH 7.0: [0, 1, 0]
"""
            log_path.write_text(log_content)
            
            df, resids = TitrationCurve.parse_log(log_path)
            
            assert resids == [20, 76, 83]
            assert len(df) == 4
            assert '20' in df.columns
            assert '76' in df.columns
            assert '83' in df.columns
    
    def test_titration_curve_parse_log_missing_header(self):
        """Test parse_log raises error for missing header"""
        from molecular_simulations.analysis.constant_pH_analysis import TitrationCurve
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'cpH.log'
            log_content = """rank=0 cpH: pH 4.0: [1, 1, 0]
rank=0 cpH: pH 7.0: [0, 0, 1]
"""
            log_path.write_text(log_content)
            
            with pytest.raises(RuntimeError, match="Could not find cpH residue ID header"):
                TitrationCurve.parse_log(log_path)
    
    def test_titration_curve_parse_log_mismatch(self):
        """Test parse_log raises error for state/residue mismatch"""
        from molecular_simulations.analysis.constant_pH_analysis import TitrationCurve
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'cpH.log'
            log_content = """cpH: resids 20  76  83
rank=0 cpH: pH 4.0: [1, 1]
"""  # Only 2 states but 3 residues
            log_path.write_text(log_content)
            
            with pytest.raises(ValueError, match="Mismatch between number of residues"):
                TitrationCurve.parse_log(log_path)
    
    def test_titration_curve_init_single_file(self):
        """Test TitrationCurve initialization with single file"""
        from molecular_simulations.analysis.constant_pH_analysis import TitrationCurve
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'cpH.log'
            log_content = """cpH: resids 20  76
rank=0 cpH: pH 4.0: [1, 0]
rank=0 cpH: pH 7.0: [0, 1]
"""
            log_path.write_text(log_content)
            
            tc = TitrationCurve(log_path)
            
            assert tc.resid_cols == ['20', '76']
            assert len(tc.df) == 2
    
    def test_titration_curve_init_multiple_files(self):
        """Test TitrationCurve initialization with multiple files"""
        from molecular_simulations.analysis.constant_pH_analysis import TitrationCurve
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log1 = Path(tmpdir) / 'cpH1.log'
            log2 = Path(tmpdir) / 'cpH2.log'
            
            log_content = """cpH: resids 20  76
rank=0 cpH: pH 4.0: [1, 0]
"""
            log1.write_text(log_content)
            log2.write_text(log_content)
            
            tc = TitrationCurve([log1, log2])
            
            assert len(tc.df) == 2  # Combined from both files


class TestHillEquation:
    """Test the Hill equation fitting functions"""
    
    def test_hill_equation_values(self):
        """Test Hill equation returns expected values"""
        from molecular_simulations.analysis.constant_pH_analysis import TitrationCurve
        
        # At pH = pKa, fraction should be 0.5
        pKa = 4.5
        n = 1.0
        
        fraction = TitrationCurve.hill_equation(pKa, pKa, n)
        assert np.isclose(fraction, 0.5, atol=0.01)
        
        # At low pH (below pKa), fraction should be > 0.5
        fraction_low = TitrationCurve.hill_equation(2.0, pKa, n)
        assert fraction_low > 0.5
        
        # At high pH (above pKa), fraction should be < 0.5
        fraction_high = TitrationCurve.hill_equation(8.0, pKa, n)
        assert fraction_high < 0.5


class TestTitrationAnalyzer:
    """Test suite for TitrationAnalyzer class"""
    
    def create_test_log(self, tmpdir):
        """Helper to create test log file with realistic data"""
        log_path = Path(tmpdir) / 'cpH.log'
        
        lines = ["cpH: resids 20  76\n"]
        
        # Generate data that follows a titration curve
        # Residue 20 is ASP (ASH=protonated, ASP=deprotonated)
        # Residue 76 is GLU (GLH=protonated, GLU=deprotonated)
        pH_values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        pKa_20 = 4.5  # Expected pKa for residue 20 (ASP)
        pKa_76 = 6.5  # Expected pKa for residue 76 (GLU)
        
        for pH in pH_values:
            n_samples = 20
            for _ in range(n_samples):
                # Probability of being protonated based on Hill equation
                p20 = 1 / (1 + 10**(pH - pKa_20))
                p76 = 1 / (1 + 10**(pH - pKa_76))
                
                # Use actual state names that protonation_mapping expects
                s20 = 'ASH' if np.random.random() < p20 else 'ASP'
                s76 = 'GLH' if np.random.random() < p76 else 'GLU'
                
                lines.append(f"rank=0 cpH: pH {pH:.1f}: ['{s20}', '{s76}']\n")
        
        log_path.write_text(''.join(lines))
        return log_path
    
    def test_titration_analyzer_init(self):
        """Test TitrationAnalyzer initialization"""
        from molecular_simulations.analysis.constant_pH_analysis import TitrationAnalyzer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self.create_test_log(tmpdir)
            
            analyzer = TitrationAnalyzer(log_path)
            
            assert analyzer.log_files == [log_path]
            assert not analyzer._analyzed
    
    def test_titration_analyzer_init_with_string(self):
        """Test TitrationAnalyzer initialization with string path"""
        from molecular_simulations.analysis.constant_pH_analysis import TitrationAnalyzer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self.create_test_log(tmpdir)
            
            analyzer = TitrationAnalyzer(str(log_path))
            
            assert len(analyzer.log_files) == 1
    
    def test_titration_analyzer_init_with_list(self):
        """Test TitrationAnalyzer initialization with list of paths"""
        from molecular_simulations.analysis.constant_pH_analysis import TitrationAnalyzer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log1 = self.create_test_log(tmpdir)
            
            # Create second log file
            log2 = Path(tmpdir) / 'cpH2.log'
            log2.write_text(log1.read_text())
            
            analyzer = TitrationAnalyzer([log1, log2])
            
            assert len(analyzer.log_files) == 2
    
    def test_titration_analyzer_run(self):
        """Test TitrationAnalyzer run method"""
        from molecular_simulations.analysis.constant_pH_analysis import TitrationAnalyzer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self.create_test_log(tmpdir)
            out_dir = Path(tmpdir) / 'output'
            
            analyzer = TitrationAnalyzer(log_path, output_dir=out_dir)
            analyzer.run(methods=['curvefit'], verbose=False)
            
            assert analyzer._analyzed
            assert hasattr(analyzer, 'fits_curvefit')
            assert analyzer.fits_curvefit is not None
    
    def test_titration_analyzer_get_results(self):
        """Test getting results after analysis"""
        from molecular_simulations.analysis.constant_pH_analysis import TitrationAnalyzer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self.create_test_log(tmpdir)
            
            analyzer = TitrationAnalyzer(log_path)
            analyzer.run(methods=['curvefit'], verbose=False)
            
            # Get results DataFrame
            results_df = analyzer.get_results(method='curvefit')
            
            # Should be a polars DataFrame with pKa values
            assert results_df is not None
            assert 'pKa' in results_df.columns
            assert 'resid' in results_df.columns
    
    def test_titration_analyzer_get_results_not_analyzed(self):
        """Test get_results returns None or raises error if not analyzed"""
        from molecular_simulations.analysis.constant_pH_analysis import TitrationAnalyzer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self.create_test_log(tmpdir)
            
            analyzer = TitrationAnalyzer(log_path)
            
            # Before running analysis, fits_curvefit should not exist
            assert not hasattr(analyzer, 'fits_curvefit') or analyzer.fits_curvefit is None
    
    def test_titration_analyzer_save_results(self):
        """Test saving results to files"""
        from molecular_simulations.analysis.constant_pH_analysis import TitrationAnalyzer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self.create_test_log(tmpdir)
            out_dir = Path(tmpdir) / 'output'
            
            analyzer = TitrationAnalyzer(log_path, output_dir=out_dir)
            analyzer.run(methods=['curvefit'], verbose=False)
            analyzer.save_results()
            
            # Check output files exist
            assert out_dir.exists()
    
    def test_titration_analyzer_repr(self):
        """Test string representation"""
        from molecular_simulations.analysis.constant_pH_analysis import TitrationAnalyzer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self.create_test_log(tmpdir)
            
            analyzer = TitrationAnalyzer(log_path)
            
            repr_str = repr(analyzer)
            assert "TitrationAnalyzer" in repr_str
            assert "not analyzed" in repr_str
            
            analyzer.run(methods=['curvefit'], verbose=False)
            repr_str = repr(analyzer)
            assert "analyzed" in repr_str


class TestAnalyzeCph:
    """Test the convenience analyze_cph function"""
    
    def create_test_log(self, tmpdir):
        """Helper to create test log file"""
        log_path = Path(tmpdir) / 'cpH.log'
        
        lines = ["cpH: resids 20\n"]
        pH_values = [3.0, 5.0, 7.0]
        
        for pH in pH_values:
            for _ in range(10):
                s = 1 if np.random.random() < 0.5 else 0
                lines.append(f"rank=0 cpH: pH {pH:.1f}: [{s}]\n")
        
        log_path.write_text(''.join(lines))
        return log_path
    
    def test_analyze_cph_basic(self):
        """Test analyze_cph convenience function"""
        from molecular_simulations.analysis.constant_pH_analysis import analyze_cph
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self.create_test_log(tmpdir)
            out_dir = Path(tmpdir) / 'output'
            
            analyzer = analyze_cph(
                log_path,
                output_dir=out_dir,
                methods=['curvefit'],
                plot=False,
                verbose=False
            )
            
            assert analyzer._analyzed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
