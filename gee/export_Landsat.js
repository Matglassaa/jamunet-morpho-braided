// specify lat/lon values 
var lon1 = 89.60040971612764;    //x1
var lat1 = 23.837723507684256;   //y1
var lon2 = 89.87094804620577;    //x2
var lat2 = 24.429258066407694;   //y2

// get center of map and center it around this point
var center_lon = (lon2+lon1)/2;
var center_lat = (lat2+lat1)/2;
var center = ee.Geometry.Point(center_lon, center_lat);
var zoom = 10;

Map.centerObject(center, zoom)

// Define polygon of interest
var geometry = ee.Geometry.Polygon([[[lon1, lat2], [lon1, lat1], 
                                     [lon2, lat1], [lon2, lat2]]]);
                                     
Map.addLayer(geometry, {}, 'geometry')
// Possible datasets
var datasets = ['JRC/GSW1_4/MonthlyHistory', 'LANDSAT/LT05/C02/T1_L2', 'LANDSAT/LE07/C02/T1', 
                'LANDSAT/LC08/C02/T1', 'LANDSAT/LC09/C02/T1', 'COPERNICUS/S1_GRD', 
                'COPERNICUS/S2_HARMONIZED', 'COPERNICUS/S2_SR_HARMONIZED', 'COPERNICUS/S3/OLCI'];

var collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
                                   .filterBounds(geometry);

var Start_period = ee.Date(collection.first()
                           .get('system:time_start'));
var End_period = ee.Date(collection.sort('system:time_start', false)
                         .first().get('system:time_start'));

var collection = collection.filterDate(Start_period, End_period);

// print information of initial collection
print('Start/end date: ', Start_period, End_period)
print('Initial collection', collection);

var vis = {min:9000, max:17000, bands: ['SR_B3', 'SR_B2', 'SR_B1'] // RGB 
          // min:7000, max:26000, bands:['SR_B5', 'SR_B4','SR_B3'] // green blue
          // min:5000, max:16000, bands:['B5', 'B4','B3'] // Landsat 9
}; 

// // ------------------------------------------------------------------ //
// uncomment following lines to add a full-tile image and initial images //

// var image4 = ee.Image('LANDSAT/LT05/C02/T1_L2/LT05_138043_19880210');
// Map.addLayer(image4.clip(geometry), vis, 'image 4 - full tile');

// var old1 = ee.Image(collection.first());
// var old2 = ee.Image(collection.toList(collection.size()).get(1));
// var old3 = ee.Image(collection.toList(collection.size()).get(2));

// Map.addLayer(old1.clip(geometry), vis,'image old 1');
// Map.addLayer(old2.clip(geometry), vis,'image old 2');
// Map.addLayer(old3.clip(geometry), vis,'image old 3');

// // ------------------------------------------------------------------ //

function makeMosaics(image) {
  var thisImage = ee.Image(image);
  var date = ee.Date(thisImage.get('system:time_start'));
  // add only hour to exclude images from other paths
  var filteredDataset = collection.filterDate(date, date.advance(1,'hour'));
  // add all image properties to the new image
  var toReturn = ee.Image(filteredDataset.mosaic()
                    .copyProperties(image,image.propertyNames())
                    );
  // add geometries
  var geometries = filteredDataset.map(function(img){
    return ee.Feature(img.geometry());
  });
  
  var mergedGeometries = geometries.union();
  return toReturn.set('system:footprint', mergedGeometries.geometry());
}

var mosaiced = collection.map(makeMosaics);

// the final collection without duplicates
var filtered_collection = mosaiced.filter(ee.Filter.contains('.geo', geometry))
                            .map(function(image){return image.clip(geometry)});

// print information of final collection
print('Final collection', filtered_collection);

// // ------------------------------------------------------------------ //
// uncomment following lines to add first three images of the filtered collection //

// var new1 = ee.Image(filtered_collection.first());
// var new2 = ee.Image(filtered_collection.toList(collection.size()).get(1));
// var new3 = ee.Image(filtered_collection.toList(collection.size()).get(2));

// Map.addLayer(new1, vis,'image new 1');
// Map.addLayer(new2, vis,'image new 2');
// Map.addLayer(new3, vis,'image new 3');

// // ------------------------------------------------------------------ //
// uncomment following lines to add slider and display one image at a time 

// renderSlider(Start_period, End_period);

// function renderSlider(Start_period, End_period) {
//         var slider = ui.DateSlider({start: Start_period, end: End_period,  // Every 5 days
//                                     period: 5, onChange: renderDateRange});
//         Map.add(slider);}

// function renderDateRange(dateRange) {
//         var image = post_filtered.filterDate(dateRange.start(), dateRange.end())
//                             .median();

//   // Define visualization
// // var vis = {min:7000, max:26000, bands: ['B4', 'B3', 'B2']}     //  Landsat - also 'B5', 'B4', 'B3'
//   var vis = {min:7000, max:26000, bands: ['SR_B5', 'SR_B4', 'SR_B3'] }
//   // var vis = {bands: ['water'], min: 0.0, max: 2.0, palette: ['ffffff', 'fffcb8', '0905ff']} // JRC
  
// var layer = ui.Map.Layer(image.clip(geometry), vis, 'Filtered');
// Map.layers().reset([layer]);
// }

// // ------------------------------------------------------------------ //

// Function to export images to Google Drive
var exportImagesToDrive = function(filtered_collection, datasetName, Start_period, End_period) {
  // Filter the collection based on the specified time period and geometry
  var filteredCollection = filtered_collection.filterBounds(geometry)
                                    .filterDate(Start_period, End_period);

  // Map over the filtered collection and export each image
  filteredCollection.evaluate(function(filtered_collection) {
    filtered_collection.features.forEach(function(feature) {
      var imageId = feature.id;
      var image = ee.Image(imageId);

      // Check if the object is actually an Image
      if (image instanceof ee.Image) {
        // Get the date of the image
        var date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');

        // Construct the export name within Earth Engine environment
        var exportName = ee.String(date);
        // Define the export parameters
        var exportParams = {
          image: image.visualize(vis),
          description: exportName.getInfo(),
          folder: 'LANDSAT/LT05/C02/T1_L2', 
          scale: 30,
          region: geometry,
          // maxPixels: 1e13
        };

        // Export the image to Google Drive
        Export.image.toDrive(exportParams);
      }
    });
  });
};

// Example usage of the function
// exportImagesToDrive(filtered_collection, 'LANDSAT/LT05/C02/T1_L2', Start_period, End_period);

// to run batch task execution copy-paste in console the following lines

// 1 // wait until all RUN are listed

// function runTaskList(){
// // var tasklist = document.getElementsByClassName('task local type-EXPORT_IMAGE awaiting-user-config');
// // for (var i = 0; i < tasklist.length; i++)
// //         tasklist[i].getElementsByClassName('run-button')[0].click();
// $$('.run-button' ,$$('ee-task-pane')[0].shadowRoot).forEach(function(e) {
//     e.click();
// })
// // }

// runTaskList(); 

// 2 // confirm all RUN

// function confirmAll() {
// // var ok = document.getElementsByClassName('goog-buttonset-default goog-buttonset-action');
// // for (var i = 0; i < ok.length; i++)
// //     ok[i].click();
// $$('ee-table-config-dialog, ee-image-config-dialog').forEach(function(e) {
//     var eeDialog = $$('ee-dialog', e.shadowRoot)[0]
//     var paperDialog = $$('paper-dialog', eeDialog.shadowRoot)[0]
//     $$('.ok-button', paperDialog)[0].click()
// })
// }

// confirmAll();

// // Loop through datasets and create charts
// for (var i = 0; i < datasets.length; i++) {
//   var dataset = datasets[i];
//   var collection = ee.ImageCollection(dataset);
//   createAndExportCharts(collection, dataset);
// }