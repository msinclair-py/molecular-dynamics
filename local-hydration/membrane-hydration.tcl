###This script does a grid-wise search of the upper
###and lower membrane, scanning local water density
###identifying regions of irregular water packing
#puts "Enter a psf file: "
#gets stdin psf
#
#puts "\nEnter a trajectory file: "
#gets stdin dcd
set psf /Scr/msincla01/YidC_Membrane_Simulation/CHARMM_GUI_Membrane/Simulation/step5_charmm2namd.psf
set dcd /Scr/msincla01/YidC_Membrane_Simulation/CHARMM_GUI_Membrane/Simulation/md1.dcd
mol new $psf
mol addfile $dcd step 100 waitfor all
animate goto start


# Identify membrane plane in 2 dimensions and avg
# z position for density search
set all [atomselect top "name P"]
set min_max [measure minmax $all]
set vec1 [lindex $min_max 1]
set vec2 [lindex $min_max 0]
set vec [vecsub $vec1 $vec2]
set x_vector [lindex $vec 0]
set y_vector [lindex $vec 1]

#need a way to calculate where to "separate"
#the definition of peri and cyto in z coord
set z [measure center $all]
set z_div [lindex $z 2]


#get z vector
proc get_z {z len mem} {
   if {$mem==0} {
      set cytoplasm [atomselect top "name P and z < $z"]
      set cyt [measure center $cytoplasm]
      set z_c [lindex $cyt 2]
      set z_low_c [expr $z_c - $len]
      set r [list $z_low_c $z_c]
   } else {
      set periplasm [atomselect top "name P and z > $z"]
      set per [measure center $periplasm]
      set z_p [lindex $per 2]
      set z_up_p [expr $z_p + $len]
      set r [list $z_p $z_up_p]

   }
   return $r
}


#first set num, the number of spaces to sample in each
#direction x and y separately and length, the x/y length
#of each grid-search box
puts "X vector is:  $x_vector"
puts "Y vector is:  $y_vector"
puts "How many grids in x/y: "
#gets stdin num
set num 15
set length [expr $x_vector / $num]

set x_start [lindex [lindex $min_max 0] 0]
set y_start [lindex [lindex $min_max 0] 1]

proc check {x x2 y y2 z z2 length mem} {
   set protein [atomselect top protein]
   set mm [measure minmax $protein]
   set px1 [lindex [lindex $mm 0] 0]
   set px2 [lindex [lindex $mm 1] 0]
   set py1 [lindex [lindex $mm 0] 1]
   set py2 [lindex [lindex $mm 1] 1]
   set pz1 [lindex [lindex $mm 0] 2]
   set pz2 [lindex [lindex $mm 1] 2]

   if {$x<$px1 && $px1<$x2 || $x<$px2 && $px2<$x2} {
      set breakx 1
   } else {
      set breakx 0
   }
 
   if {$y<$py1 && $py1<$y2 || $y<$py2 && $py2<$y2} {
      set breaky 1
   } else {
      set breaky 0
   }

   if {$z<$pz1 && $pz1<$z2 || $z<$pz2 && $pz2<$z2} {
      set breakz 1
   } else {
      set breakz 0
   }

   if {$breakx==1 || $breaky==1 || $breakz== 1} {
      set half_length [expr $length * 2 / 3]
      if {$x2 > $px1 || $y2 > $py1} {
         set xnew $x
         set x2new [expr $x + $half_length]
         set ynew $y
         set y2new [expr $y + $half_length]
         if {$mem==1} {
            set znew $z
            set z2new [expr $z + $half_length]
         } else {
            set znew [expr $z2 - $half_length]
            set z2new $z2
         }
      } elseif {$px2 > $x || $py2 > $y} {
         set xnew [expr $x2 - $half_length]
         set x2new $x2
         set ynew [expr $y2 - $half_length]
         set y2new $y2
      }
      if {$mem==1} {
         set znew $z
         set z2new [expr $z + $half_length]
      } else {
         set znew [expr $z2 - $half_length]
         set z2new $z2
      }

   } else {
      set xnew $x
      set x2new $x2
      set ynew $y
      set y2new $y2
      set znew $z
      set z2new $z2
      set half_length $length
   }
  
   set vol [expr $half_length ** 3]
   set rtrn [list $xnew $x2new $ynew $y2new $znew $z2new $vol]
   return $rtrn
}


#define a process to export much of the calculation from the for loop
proc density {ii jj len xs ys z_low z_up mem} {
   set x_s [expr $ii * $len + $xs]
   set y_s [expr $jj * $len + $ys]
   set x_great [expr $x_s + $len]
   set y_great [expr $y_s + $len]
  
   set pchk [check $x_s $x_great $y_s $y_great $z_low $z_up $len $mem]
   set volume [lindex $pchk 6]
   set x [lindex $pchk 0]
   set x2 [lindex $pchk 1]
   set y [lindex $pchk 2]
   set y2 [lindex $pchk 3]
   set z [lindex $pchk 4]
   set z2 [lindex $pchk 5]
  
   set sel [atomselect top "name OH2 and ((x>$x and x<$x2) and (y>$y and y<$y2) and (z>$z and z<$z2))"]
   set reslist [$sel get resid]
   set waters [lsort -unique $reslist]
   set num_water [llength $waters]
   set dens [expr $num_water * 10000 / ($volume * 6.0221409)]
   if {$dens > 70 || $dens < 30} {
      set fo [open "errorlog$ii.txt" w]
      puts $fo "$x $x2 $y $y2 $z $z2 $num_water $volume $dens"
      close $fo
   }
   return $dens
}

#iterate through for cytoplasm and then periplasm
set lf [molinfo top get numframes]

for {set k 0} {$k < $lf} {incr k} {
   animate goto $k
   set fcyt [open "cytoplasm$k.txt" w]
   set fper [open "periplasm$k.txt" w]
   for {set i 0} {$i < $num} {incr i} {
      for {set j 0} {$j < $num} {incr j} {
         set x [expr $i * $length + $x_start]
         set rx [expr {round($x)}]
         set y [expr $j * $length + $y_start]
         set ry [expr {round($y)}]
         set z [get_z $z_div $length 0]
         set z1 [lindex $z 0]
         set z2 [lindex $z 1]
         set d [density $i $j $length $x_start $y_start $z1 $z2 0]
         puts $fcyt "$rx $ry $d"
      }
   }
   close $fcyt

   for {set i 0} {$i < $num} {incr i} {
      for {set j 0} {$j < $num} {incr j} {
         set x [expr $i * $length + $x_start]
         set rx [expr {round($x)}]
         set y [expr $j * $length + $y_start]
         set ry [expr {round($y)}]
         set z [get_z $z_div $length 1]
         set z1 [lindex $z 0]
         set z2 [lindex $z 1]
         set d [density $i $j $length $x_start $y_start $z1 $z2 1]
         puts $fper "$rx $ry $d"
      }
   }
   close $fper
}

exit
