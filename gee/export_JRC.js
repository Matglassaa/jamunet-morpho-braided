// set geometry (choose among training, validation, or testing)
var geometry = testing; //.filterMetadata('reach_id', 'equals', 28); //change features depending on the area to be exported
print('Training dataset locations', geometry.size()); 
          
// load collection
var collection = ee.ImageCollection('JRC/GSW1_4/MonthlyHistory')
                                   .filterBounds(geometry);
                    
// start date as the system time of the first image
// var Start_period = ee.Date(collection.first()
//                           .get('system:time_start'));

// earlier images are completely empty
var Start_period = ee.Date('1987-12-01');

// end date as the system time of the last image
var End_period = ee.Date(collection.sort('system:time_start', false)
                        .first().get('system:time_start')).advance(1, 'day');

// var End_period = ee.Date('1999-02-02');
                     
// filter collection in the selected period only   
var collection = collection.filterDate(Start_period, End_period);

print('Start:', Start_period)
print('End', End_period)
print('Dataset size', collection.size())

// Visualization parameters for the water band - not needed
// var waterVis = {
//   min: 0, 
//   max: 2, 
//   bands: ['water'],
//   palette: ['white', 'green', 'blue'] // Color palette for visualization
// };

// Function to export images to Google Drive
var exportImagesToDrive = function(collection, datasetName, reach_id, use, Start_period, End_period) {
  // Filter the collection based on the specified time period and geometry
  var filteredCollection = collection.filterBounds(use)
                                    .filterDate(Start_period, End_period);

  // Map over the filtered collection and export each image
  filteredCollection.evaluate(function(collection) {
    collection.features.forEach(function(feature) {
      var geometry_reach = use.filterMetadata('reach_id', 'equals', reach_id);
      var use_str = geometry_reach.first().get('use').getInfo();
      var imageId = feature.id;
      var image = ee.Image(imageId).clip(geometry_reach);
      var str_reach_id = ee.String(ee.Number(reach_id).int());

      // Check if the object is actually an Image
      if (image instanceof ee.Image) {
        // Get the date of the image
        var date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');
        // var task = ee.String('training'); // change based on area selected;

        // Construct the export name within Earth Engine environment
        var exportName = ee.String(datasetName.replace(/\//g, "_")).cat('_').cat(date)
                                              .cat('_').cat(use_str)
                                              .cat('_r').cat(str_reach_id);
        // var exportName = temp_exportName.replace(/\//g, "_");
                                              
        var exportFolder = ee.String(datasetName.replace(/\//g, "_"))
                                                .cat('_').cat(use_str)
                                                .cat('_r').cat(str_reach_id); 

        // Define the export parameters
        var exportParams = {
          image: image, //.visualize(waterVis) - remove this to get grayscale images
          description: exportName.getInfo(),
          folder: exportFolder.getInfo(), 
          scale: 60,
          region: geometry_reach,
        };

        // Export the image to Google Drive
        Export.image.toDrive(exportParams);
      }
    });
  });
};

var list_reaches = ee.List.sequence(ee.Number(1), training.size());

// export single reach
exportImagesToDrive(collection, 'JRC_GSW1_4_MonthlyHistory', 24, training, Start_period, End_period);

// // loop through reach_id to export all reaches at the same time - not working, the process crashes!
// for (var i = 1; i <= training.size().getInfo(); i++) {
//   exportImagesToDrive(collection, 'JRC_GSW1_4_MonthlyHistory', i, training, Start_period, End_period);
// }

// for (var i = 1; i <= 3; i++) {
//   exportImagesToDrive(collection, 'JRC_GSW1_4_MonthlyHistory', i, training, Start_period, End_period);
// }

// ----------------------------------------------------------------------- //

// // to automatically export all images without manually clicking on each run, follow these instructions
// run batch task execution in console 

// 1 // wait until all RUN are listed

// press F12 to open console and paste the following code and run it

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

// in console paste the following code and run it
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