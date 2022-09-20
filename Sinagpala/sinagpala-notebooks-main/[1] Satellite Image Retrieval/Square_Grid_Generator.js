const { squareGrid, polygon } = require("@turf/turf")
const { writeFileSync } = require('fs')

// Replace with your json file
const taguig = require('./taguig_polygon.json')['features'][0]['geometry']['coordinates']

//[minX, minY, maxX, maxY] of your chosen city
var bbox = [121.02220566, 14.45131156 ,121.12215097, 14.56213046];

var cellSide = 76; //size of each square
var poly = polygon(taguig)
var options = {units: 'meters', mask: poly};
var sg = squareGrid(bbox, cellSide, options);

var dictstring = JSON.stringify(sg);
writeFileSync("taguig_boxes.json", dictstring);
