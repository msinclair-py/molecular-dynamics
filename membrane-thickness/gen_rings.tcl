proc getThicc {low high} {
	set top [atomselect top "(name P and within $high of protein) and (not within $low of protein) and z>0"]
	set bot [atomselect top "(name P and within $high of protein) and (not within $low of protein) and z<0"]
	set t [measure center $top]
	set b [measure center $bot]
	puts "top: $t\nbot: $b"

	set thick [lindex [vecsub [measure center $top] [measure center $bot]] 2]
	return $thick
}


#load traj and align to protein
source loadtraj.tcl

set width 3
set nf [molinfo top get numframes]
set fo [open "thickness.txt" w]

#set number of rings to evaluate
set phosphates [atomselect top "name P"]

set sys [measure minmax $phosphates]
set x [lindex [vecsub [lindex $sys 1] [lindex $sys 0]] 0]

#number of rings should be less than would fit so space uninhabited by lipids isn't sampled
set num [expr {int([expr $x / (2*$width)])} - 2]


#establish ring selections
for {set fr 0} {$fr < $nf} {incr fr} {
	animate goto $fr
	for {set i 1} {$i <= $num} {incr i} {
		set end [expr {$i * 3}]
		set start [expr {$end - 3}]
		set thicc [getThicc $start $end]
		puts $fo "$fr $i $thicc"
	}
}

exit
