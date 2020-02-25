source loadtraj.tcl
set pore "protein and resid 424 to 474"
source alignment.tcl
set nf [molinfo top get numframes]

proc poreVector {top bot} {
	set v2 [measure center $top]
	set v1 [measure center $bot]
	return [vecnorm [vecsub $v2 $v1]]
}


proc formatSelection {x} {
	if {$x < 0} {
		set alter [expr -1*$x]
		set xf "+ ${alter}"
	} else {
		set xf "- $x"
	}
	return $xf
}


proc avgPosition {v1 v2} {
	set vec [vecscale 0.5 [vecadd $v1 $v2]]
	set x [lindex $vec 0]
	set y [lindex $vec 1]
	return [list $x $y]
}


proc defineCylinder {vector up dwn} {
	#need vector components
	set vx [lindex $vector 0]
	set vy [lindex $vector 1]
	set vz [lindex $vector 2]
	set u [measure center $up]
	set d [measure center $dwn]
	set uz [lindex $u 2]
	set dz [lindex $d 2]

	#need x/y offset since the com of the protein segment isn't at the origin
	set avg [avgPosition $u $d]
	set xoff [lindex $avg 0]
	set yoff [lindex $avg 1]

	#using projection identities, get the z coefficient for x/y
	set theta [expr {atan([expr ($vy/$vx)])}]
	set phi [expr {atan([expr (($vx**2+$vy**2)**.5/$vz)])}]
	set xcomp [expr [expr {tan ($phi)}] * [expr {cos ($theta)}]]
	set ycomp [expr [expr {tan ($phi)}] * [expr {sin ($theta)}]]

	#atomselection text
	set xc [formatSelection $xcomp]
	set yc [formatSelection $ycomp]
	set xo [formatSelection $xoff]
	set yo [formatSelection $yoff]

	set select "name OH2 and ((x ${xc}*z ${xo})^2+(y ${yc}*z ${yo})^2 < 100) and z < ${uz} and z > ${dz}"
	return $select
}


proc getDensity {sel fr} {
	global outfile
	set wat [atomselect top ${sel}]
	foreach resid [$wat get {x y z}] {
		set res [lindex $resid 2]
		puts $outfile "$fr $res"
	}
}


proc waterProfile {nf top bot up down all} {
	for {set i 0} {$i < $nf} {incr i} {
		$top frame $i
		$bot frame $i
		$up frame $i
		$down frame $i
		$all frame $i

		set vector [poreVector $top $bot]
		set sel [defineCylinder $vector $up $down]
		getDensity $sel $i
	}
}


#####********************#####
###Body of code starts here###
#####********************#####
puts "aligning structure....."
align_sel $pore
set top [atomselect top "protein and resid 467 to 471"]
set bot [atomselect top "protein and resid 457 to 461"]
set up [atomselect top "protein and resid 470"]
set down [atomselect top "protein and resid 428"]
set all [atomselect top "all"]
set outfile [open "pore_water_density.txt" w]

waterProfile $nf $top $bot $up $down $all
close $outfile

exit
