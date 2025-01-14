proc iterateTrajectory {} {
	global fo
	for {set i 0} {$i < $numframes} {incr i} {
		$wats frame $i
		set res [lsort -unique [$wats get resid]]
		foreach r $res {
			set zpos [lindex [$r get {x y z}] 2]
			puts $fo "$i $res $zpos"
		}
	}
}


source loadtraj.tcl

set nf [molinfo top get numframes]
set fo [open "surfwaters.txt" w]

set wats [atomselect top "segname TIP3 and within 5 of ((protein and resid xx to yy) or (protein and resid xx to yy))"]
#set helix1 [atomselect top "protein and resid xx to yy"]
#set helix2 [atomselect top "protein and resid xx to yy"]

exit
