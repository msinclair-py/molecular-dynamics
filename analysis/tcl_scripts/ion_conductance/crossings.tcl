proc getBoundary {top bot} {
	set t [lindex [measure center $top] 2]
	set b [lindex [measure center $bot] 2]
	return [list $t $b]
}


proc whichSlab {t b z} {
	if {$z < $b} {
		return 0
	} elseif {$z > $t} {
		return 2
	} else {
		return 1
	}
}


proc mapIons {f t b} {
	global outfile
	global numpots
	for {set j 1} {$j <= $numpots} {incr j} {
		set pot [atomselect top "name POT and resid $j"]
		set z [lindex [measure center $pot] 2]
		set slab [whichSlab $t $b $z]
		puts $outfile "$f $j $slab"
	}
}

source loadtraj.tcl

set nf [molinfo top get numframes]
set top [atomselect top "name P and z>0"]
set bot [atomselect top "name P and z<0"]
set outfile [open "ionpositions.txt" w]

set numpots 664

for {set i 0} {$i < $nf} {incr i} {
	$top frame $i
	$bot frame $i
	set bounds [getBoundary $top $bot]
	set tz [lindex $bounds 0]
	set bz [lindex $bounds 1]
	mapIons $i $tz $bz
}

exit
