proc getRes {selection} {
	set res [lsort -unique [$selection get resid]]
	return $res
}


proc getType {resid} {
	set residue [atomselect top "segname MEMB and resid $resid"]
	set type [lsort -unique [$residue get resname]]
	return $type
}


proc getVec {sel1 sel2 sel3} {
	set c1 [$sel1 get {x y z}]
	set c2 [$sel2 get {x y z}]
	set c3 [$sel3 get {x y z}]

	set x1 [lindex $c1 0]
	set x2 [lindex $c2 0]
	set x3 [lindex $c3 0]

	set y1 [lindex $c1 1]
	set y2 [lindex $c2 1]
	set y3 [lindex $c3 1]

	set z1 [lindex $c1 2]
	set z2 [lindex $c2 2]
	set z3 [lindex $c3 2]

	set v1 [vecnorm [vecsub $c1 $c2]]
	set v2 [vecnorm [vecsub $c2 $c3]]

	set vec [vecnorm [vecadd $v1 $v2]]
	return $vec
}


proc getTails {resid} {
	set sn1_t [atomselect top "name C22 and segname MEMB and resid $resid"]
	set sn1_m [atomselect top "name C28 and segname MEMB and resid $resid"]
	set sn1_b [atomselect top "name C215 and segname MEMB and resid $resid"]

	set sn2_t [atomselect top "name C32 and segname MEMB and resid $resid"]
	set sn2_m [atomselect top "name C38 and segname MEMB and resid $resid"]
	set sn2_b [atomselect top "name C315 and segname MEMB and resid $resid"]

	set sn1_vec [getVec $sn1_t $sn1_m $sn1_b]
	set sn2_vec [getVec $sn2_t $sn2_m $sn2_b]

	return [list $sn1_vec $sn2_vec]
}


proc iterateFrames {lipids control} {
	global nf
	global fo

	for {set i 0} {$i < $nf} {incr i} {
		$lipids frame $i
		$control frame $i

		set lres [getRes $lipids]
		set cres [getRes $control]

		foreach res $lres {
			set type [getType $res]
			set tails [getTails $res]
			puts "6"
			set t1 [lindex $tails 0]
			set t2 [lindex $tails 1]
			puts "outputting"

			puts $fo "$i $res $type $t1 $t2"
		}
	}
}

source loadtraj.tcl
set nf [molinfo top get numframes]
set fo [open "lipidtilts.txt" w]

set lipids [atomselect top "segname MEMB and within 5 of protein"]
set control [atomselect top "segname MEMB and not within 10 of protein"]

iterateFrames $lipids $control

exit
