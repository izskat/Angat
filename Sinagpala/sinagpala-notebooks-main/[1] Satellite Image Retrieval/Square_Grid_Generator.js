const { squareGrid, polygon } = require("@turf/turf")
const { writeFileSync } = require('fs')

// Replace with your json file
const taguigjson = require('./taguig_polygon.json')['features'][0]['geometry']['coordinates']
const makatijson = require('./makati_polygon.json')['features'][0]['geometry']['coordinates']
const laspinasjson = require('./laspinas_polygon.json')['features'][0]['geometry']['coordinates']

//[minX, minY, maxX, maxY] of your chosen city
var bboxT = [121.022206,14.451312,121.122151,14.56213];
var bboxM = [120.998771,14.529634,121.067503,14.579432]
var bboxL = [120.943353,14.349185,121.024234,14.501115]

function getboxes(json, bbox, city) {
    var cellSide = 76; //size of each square
    var poly = polygon(json)
    var options = {units: 'meters', mask: poly};
    var sg = squareGrid(bbox, cellSide, options);
    
    var dictstring = JSON.stringify(sg);
    writeFileSync( city + "_boxes.json", dictstring);
}

getboxes(taguigjson, bboxT, "taguig")
getboxes(makatijson, bboxM, "makati")
getboxes(laspinasjson, bboxL, "laspinas")