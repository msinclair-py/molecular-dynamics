proc getThicc {low high} {
	set tt "(name P and within $high of protein) and (not within $low of protein) and z>0"
	set bb "(name P and within $high of protein) and (not within $low of protein) and z<0"
	set top [atomselect top $tt]
	set bot [atomselect top $bb]

	if {
		[llength [$top get resid]] == 0 || [llength [$bot get resid]] == 0
	} then {
		set thick "NaN"
	} else {
		set t [measure center $top]
		set b [measure center $bot]
		set thick [lindex [vecsub [measure center $top] [measure center $bot]] 2]
	}

	return $thick
}


proc aligner {nf all} {
	set ref [atomselect top "segname MEMB" frame 0]
	set sel [atomselect top "segname MEMB"]
	
	for {set i 0} {$i < $nf} {incr i} {
		$sel frame $i
		$all frame $i
		set tmatrix [measure fit $sel $ref]
		$all move $tmatrix
	}
}


#load traj and align to protein
mol new top6.psf
mol addfile top6_100ns_125ps.dcd waitfor all

set width 3
set nf [molinfo top get numframes]
set fo [open "top6_thickness.txt" w]

#set number of rings to evaluate
set phosphates [atomselect top "name P"]

set sys [measure minmax $phosphates]
set x [lindex [vecsub [lindex $sys 1] [lindex $sys 0]] 0]

#number of rings should be less than would fit so space uninhabited by lipids isn't sampled
set num [expr {int([expr $x / (2*$width)])} - 2]

#selections to recenter membrane to avoid drift issues with calculations
set all [atomselect top "all"]

aligner $nf $all

#establish ring selections
for {set fr 0} {$fr < $nf} {incr fr} {
	animate goto $fr
	puts "Frame $fr"
	for {set i 1} {$i <= $num} {incr i} {
		set end [expr {$i * 3}]
		set start [expr {$end - 3}]
		set thicc [getThicc $start $end]
		puts $fo "$fr $i $thicc"
	}
}

exit
