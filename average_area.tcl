puts "Enter a psf file: "
gets stdin psf
puts "\nEnter a trajectory file: "
gets stdin traj
puts "\nEnter a stride: "
gets stdin stride

mol new $psf
mol addfile $traj step $stride waitfor all
animate goto start

set fo [open "avgarea.txt" w+]

proc area {a b} {
   return [expr $a * $b]
}

proc getvec {fram} {
   animate goto $fram
   set memb [atomselect top "name P"]
   set meas [measure minmax $memb]
   set vec1 [lindex $meas 1]
   set vec2 [lindex $meas 0]
   set x [expr [lindex $vec1 0] - [lindex $vec2 0]]
   set y [expr [lindex $vec1 1] - [lindex $vec2 1]]
   return [area $x $y]
}

set lf  [molinfo top get numframes]

for {set i 0} {$i < $lf} {incr i} {
   set ar [getvec $i]
   puts $fo "$i $ar"
}

quit
