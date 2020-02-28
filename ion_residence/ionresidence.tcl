proc getResidence {fr residues} {
	global fo
	foreach res $residues {
		set pot [lsort -unique [[atomselect top "name POT and within 4 of protein and resid $res"] get resid]]
		set count [llength $pot]
		puts $fo "$fr $res $count"
	}
}


source loadtraj.tcl
set fo [open "residences.txt" w]
set nf [molinfo top get numframes]

set ions [atomselect top "name POT and within 10 of protein"]

# pore was 430 to 474; too much ion residence on extracellular region
# outer was 325 to 420; too much ion residence on the extracellular region
set pore [atomselect top "protein and resid 430 to 467"] 
set outer [atomselect top "protein and ((resid 335 to 360) or (resid 387 to 413))"]

set pore_residues [lsort -unique [$pore get resid]]
set outer_residues [lsort -unique [$outer get resid]]

for {set i 0} {$i < $nf} {incr i} {
	animate goto $i
	getResidence $i $pore_residues
	getResidence $i $outer_residues
}

exit
