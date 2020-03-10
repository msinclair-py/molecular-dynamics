proc alignMembrane {} {
	global nf
	set ref [atomselect top "segname MEMB" frame 0]
	set compare [atomselect top "segname MEMB"]
	
	for {set i 0} {$i < $nf} {incr i} {
		$compare frame $i
		set trans_mat [measure fit $compare $ref]
		$compare move $trans_mat
	}
}


proc gridSpace {sel} {
	global grid
	set mm [measure minmax $sel]
	set x1 [lindex [lindex $mm 0] 0]
	set y1 [lindex [lindex $mm 0] 1]
#	set z [lindex [measure center $sel] 2]

	#xlen == ylen due to namd settings
	set xlen [expr {[lindex [lindex $mm 1] 0] - $x1}]
	set numgrids [expr {$xlen/$grid}]
	#offset the initial x position such that the N grids are centered
	set displace [expr {int($xlen) % $grid * $grid/2}]
	set x [expr {$x1 + $displace}]
	set y [expr {$y1 + $displace}]
	return [list $x $y $numgrids]
#	return [list $x $y $z $numgrids]
}


proc gridSearch {specs op} {
	global fo
	global grid
	set x [lindex $specs 0]
	set y [lindex $specs 1]
	set N [lindex $specs 2]
#	set z [lindex $specs 2]
#	set N [lindex $specs 3]

	#ensure z1 is always lower than z2 for consistency across bilayer
#	if {$z < 0} {
#		set op "<0"
#		set z1 [expr {$z1 - $grid}]
#		set z2 $z
#	} else {
#		set op ">0"
#		set z1 $z
#		set z2 [expr {$z1 + $grid}]
#	}

	#do the gridsearch
	for {set i 1} {$i <= $N} {incr i} {
		set x1 [expr {$x + $grid * $i}]
		set x2 [expr {$x + $grid * ($i + 1)}]

		for {set j 1} {$j <= $N} {incr j} {
			set y1 [expr {$y + $grid * $j}]
			set y2 [expr {$y + $grid * ($j + 1)}]
			set gr [atomselect top "name POT and within 4 of name P and x>$x1 and x<$x2 and y>$y1 and y<$y2 and z$op"]
			set num [llength [$gr get resid]]
			puts $fo "$i $j $num"
		}		
	}
}


proc iterateFrames {} {
	global nf
	set all [atomselect top "all"]
	set memb [atomselect top "name P"]
	for {set i 0} {$i < $nf} {incr i} {
		$all frame $i
		set g [gridSpace $memb]
		gridSearch $g ">0"
		gridSearch $g "<0"
	}
}


source loadtraj.tcl
set nf [molinfo top get numframes]
set fo [open "iondensity.txt" w]
#use a 5A grid
set grid 5

alignMembrane
iterateFrames
