// Load the parcels FeatureCollection
var parcelas = ee.FeatureCollection('projects/resag-1/assets/Cereal_Comarca_V')

// Filter parcels for 'Convencional' management
var convencional = ee.FeatureCollection(parcelas)
    .filter(ee.Filter.eq('Manejo','Convencional'));
// Filter parcels for 'Conservacion' management
var conservacion = ee.FeatureCollection(parcelas)
    .filter(ee.Filter.eq('Manejo','Conservacion'));
 

// Function to mask clouds using the SCL band
function maskclouds_scl(imagen) {
  var scl = imagen.select('SCL');
  // Select vegetation, soil, water, and snow classes
  var veg = 4;
  var soil = 5;
  var water = 6;
  var snow = 11;
  // Mascara
  var mask = scl.eq(veg).or(scl.eq(soil)).or(scl.eq(water)).or(scl.eq(snow));
  var properties = imagen.propertyNames();
  var imagen_mask = ee.Image(imagen.updateMask(mask).copyProperties(imagen,properties));
  return imagen_mask;
}



// Load Sentinel-2 Level-2A imagery collection
var S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
// Filter the imagery collection by date and location, apply cloud mask, and select bands
var dataset = S2.filter(ee.Filter.date('2022-07-01', '2023-06-30'))
                // filtro inicialmente por nubosidad
                  //.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',10))
                  .filterBounds(parcelas)
                  .map(maskclouds_scl)
                  .select('B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12');
print(dataset)                  

// Function to compute statistics for parcels under 'Convencional' management

var stats = dataset.map(function(image) {
  return convencional.map(function(f){
    var mean = image.reduceRegion({reducer: ee.Reducer.mean(),geometry: f.geometry(),scale: 20});
    var std = image.reduceRegion({reducer: ee.Reducer.stdDev(),geometry: f.geometry(),scale: 20});
    return f.set({
      'date': image.date().format(),
      // MEDIA
      'B2_mean': mean.get('B2'),'B6_mean': mean.get('B6'),
      'B3_mean': mean.get('B3'),'B7_mean': mean.get('B7'),
      'B4_mean': mean.get('B4'),'B8_mean': mean.get('B8'),
      'B5_mean': mean.get('B5'),'B8A_mean': mean.get('B8A'),
      'B11_mean': mean.get('B11'),'B12_mean': mean.get('B12'),
      // DESVIACION ESTANDAR
      'B2_std': std.get('B2'),'B6_std': std.get('B6'),
      'B3_std': std.get('B3'),'B7_std': std.get('B7'),
      'B4_std': std.get('B4'),'B8_std': std.get('B8'),
      'B5_std': std.get('B5'),'B8A_std': std.get('B8A'),
      'B11_std': std.get('B11'),'B12_std': std.get('B12'),
    })
  })
})
.flatten()
// Filter out parcels with missing band values
.filter(ee.Filter.neq('B2_mean', null))
.filter(ee.Filter.neq('B3_mean', null)).filter(ee.Filter.neq('B4_mean', null))
.filter(ee.Filter.neq('B5_mean', null)).filter(ee.Filter.neq('B6_mean', null))
.filter(ee.Filter.neq('B7_mean', null)).filter(ee.Filter.neq('B8_mean', null))
.filter(ee.Filter.neq('B8A_mean', null)).filter(ee.Filter.neq('B11_mean', null))
.filter(ee.Filter.neq('B12_mean', null))
// DESVIACION ESTANDAR
.filter(ee.Filter.neq('B2_std', null))
.filter(ee.Filter.neq('B3_std', null)).filter(ee.Filter.neq('B4_std', null))
.filter(ee.Filter.neq('B5_std', null)).filter(ee.Filter.neq('B6_std', null))
.filter(ee.Filter.neq('B7_std', null)).filter(ee.Filter.neq('B8_std', null))
.filter(ee.Filter.neq('B8A_std', null)).filter(ee.Filter.neq('B11_std', null))
.filter(ee.Filter.neq('B12_std', null));
// Export statistics to Google Drive
Export.table.toDrive({
  collection: stats,
  description: 'Comarca_V_bandas_S2_convencional',
  fileFormat: 'CSV',
  folder: 'Proyecto_ReSag'
}); 

// Repeat the same process for 'Conservacion' management
var stats = dataset.map(function(image) {
  return conservacion.map(function(f){
    var mean = image.reduceRegion({reducer: ee.Reducer.mean(),geometry: f.geometry(),scale: 20});
    var std = image.reduceRegion({reducer: ee.Reducer.stdDev(),geometry: f.geometry(),scale: 20});
    return f.set({
      'date': image.date().format(),
      // MEDIA
      'B2_mean': mean.get('B2'),'B6_mean': mean.get('B6'),
      'B3_mean': mean.get('B3'),'B7_mean': mean.get('B7'),
      'B4_mean': mean.get('B4'),'B8_mean': mean.get('B8'),
      'B5_mean': mean.get('B5'),'B8A_mean': mean.get('B8A'),
      'B11_mean': mean.get('B11'),'B12_mean': mean.get('B12'),
      // DESVIACION ESTANDAR
      'B2_std': std.get('B2'),'B6_std': std.get('B6'),
      'B3_std': std.get('B3'),'B7_std': std.get('B7'),
      'B4_std': std.get('B4'),'B8_std': std.get('B8'),
      'B5_std': std.get('B5'),'B8A_std': std.get('B8A'),
      'B11_std': std.get('B11'),'B12_std': std.get('B12'),
    })
  })
})
.flatten()
// MEDIA
.filter(ee.Filter.neq('B2_mean', null))
.filter(ee.Filter.neq('B3_mean', null)).filter(ee.Filter.neq('B4_mean', null))
.filter(ee.Filter.neq('B5_mean', null)).filter(ee.Filter.neq('B6_mean', null))
.filter(ee.Filter.neq('B7_mean', null)).filter(ee.Filter.neq('B8_mean', null))
.filter(ee.Filter.neq('B8A_mean', null)).filter(ee.Filter.neq('B11_mean', null))
.filter(ee.Filter.neq('B12_mean', null))
// DESVIACION ESTANDAR
.filter(ee.Filter.neq('B2_std', null))
.filter(ee.Filter.neq('B3_std', null)).filter(ee.Filter.neq('B4_std', null))
.filter(ee.Filter.neq('B5_std', null)).filter(ee.Filter.neq('B6_std', null))
.filter(ee.Filter.neq('B7_std', null)).filter(ee.Filter.neq('B8_std', null))
.filter(ee.Filter.neq('B8A_std', null)).filter(ee.Filter.neq('B11_std', null))
.filter(ee.Filter.neq('B12_std', null));
// Export
Export.table.toDrive({
  collection: stats,
  description: 'Comarca_V_bandas_S2_conservacion',
  fileFormat: 'CSV',
  folder: 'Proyecto_ReSag'
}); 
