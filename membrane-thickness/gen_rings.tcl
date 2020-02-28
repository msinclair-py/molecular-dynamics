puts "Enter a structure file: "
gets stdin struc
puts "\nEnter a trajectory: "
gets stdin traj
puts "\nEnter a ring width in A: "
gets stdin width

#load structure
mol new $struc
mol addfile $traj waitfor all
animate goto end

#get rough average and set number of rings to evaluate
set rough [atomselect top "name P"]
$rough writepdb netavg.pdb

set sys [measure minmax $rough]
set x [lindex [vecsub [lindex $sys 1] [lindex $sys 0]] 0]
set num [expr {int([expr $x / (2*$width)])} - 2]
#the -2 in num is arbitrary and just to prevent undersampling of
#faraway lipids

set ring1 [atomselect top "name P and within $width of protein"]
$ring1 writepdb ring1.pdb

for {set i 2} {$i < $num} {incr i} {
   set low [expr (($i -1) * $width)]
   set high [expr ($i * $width)]
   set ring [atomselect top "(name P and within $high of protein) and (not within $low of protein)"]
   $ring writepdb ring$i.pdb
}


exit
