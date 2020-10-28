// Display
    
// obstacles
translate([5, 2.5, -1.5+6])
    cube([10,5,3], center=true);
    
translate([10, 2.5, -1.5+12])
    cube([10,5,3], center=true);
  
  
// side walls
translate([7.5, 2.5, -0.5])
    cube([17,5,1], center=true);
    
translate([7.5, 2.5, -0.5+16])
    cube([17,5,1], center=true);
    
translate([0.5+15, 2.5, 7.5])
    cube([1,5,16], center=true);
    
translate([-0.5, 2.5, 7.5])
    cube([1,5,16], center=true);
    
// bottom
translate([7.5, 0, 7.5])
    cube([17,2,17], center=true);
  