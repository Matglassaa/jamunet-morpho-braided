# JamUNet: predicting the morphological changes of braided sand-bed rivers with deep learning


<img src="Figure 1.png" width="1000" alt="Brahmaputra-Jamuna River">


<table>
  <tr>
    <td>
      <img src=".\images\1994-01-25.png" width="1000" alt="Brahmaputra-Jamuna River">
    </td>
    <td>
      <p style="font-size: 16px;">
        This repository stores the data, code, and other files necessary for the completion of the Master's thesis of <a href="https://nl.linkedin.com/in/antonio-magherini-4349b2229">Antonio Magherini</a>, student of the MSc Civil Engineering program - Hydraulic Engineering track, with a specialisation in River Engineering 
        at the <a href="https://www.tudelft.nl/citg">Faculty of Civil Engineering and Geosciences</a> of Delft University of Technology (TU Delft).
      </p>
      <p style="font-size: 16px;">
        The thesis was carried out in collaboration with <a href="https://www.deltares.nl/en">Deltares</a>. The manuscript can be found at <a href="https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348">TU Delft repository</a>.
      </p>
      <p style="font-size: 16px;">
        For any information, feel free to contact the author at: <a href="mailto:antonio.magherini@epfl.ch"><em>antonio.magherini@epfl.ch</em></a>.
      </p>
      <p style="margin-top: 100px;">
        <em>The image represents the Brahmaputra-Jamuna River at the border between India and Bangladesh. The image was taken on January 25, 1994. It was retrieved from <a href="https://earthengine.google.com/">Google Earth Engine</a> <a href="https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2">USGS Landsat 5 collection</a>.</em>
      </p>
    </td>
  </tr>
</table>

---

## Repository structure

The structure of this repository is the following:
- <code>benchmarks</code>, contains modules and notebooks of the benchmark models used for comparison;
- <code>data</code>, contains raw data (satellite images, river variables);
- <code>gee</code>, contains the scripts as <code>.js</code> files necessary for exporting the images from Google Earth Engine; (work in progress)
- <code>images</code>, contains the images shown in the thesis report and other documents; (to be added soon)
- <code>model</code>, contains the modules and noteboooks with the deep-learning model;
- <code>other</code>, contains documents, images, and other files used during the project;
- <code>postprocessing</code>, contains the modules used for the data postprocessing;
- <code>preliminary</code>, contains the notebooks with the preliminary data analysis, satellite image visualization, preprocessing steps, and other examples;
- <code>preprocessing</code>, contains the modules used for the data preprocessing.

---

## Install dependencies

<code>braided.yml</code> is the environment file with all dependencies, needed to run the notebooks.

To activate the environment follow these steps:

- make sure to have the file <code>braided.yml</code> in your system (for Windows users, store it in <code>C:\Windows\System32</code>);
- open the anaconda prompt;
- run <code>conda env create -f braided.yml</code>;
- verify that the environment is correctly installed by running <code>conda env list</code> and checking the environment exists;
- activate the environment by running <code>conda activate braided</code>;
- deactivate the environment by running <code>conda deactivate</code>;

---

## Cite

Please cite the [Master thesis](https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348) as:

```
@mastersthesis{magherini2024,
author = {Magherini, A.},
title = {{JamUNet: predicting the morphological changes of braided sand-bed rivers with deep learning}},
school = {{Delft University of Technology}},
year = {2024},
month = {10},
howpublished = {\url{https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348}}
}
```
<!-- 
<p align="center" style="margin-top: 1px;"> 
    <img src=".\images\1994-01-25.png" width="400"> 
</p>

<p align="center">
    <em>Brahmaputra-Jamuna River at the border between India and Bangladesh. The image was retrieved<br>from <a href="https://earthengine.google.com/">Google Earth Engine</a> <a href="https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2">USGS Landsat 5 collection</a>. The image was taken on January 25, 1994.</em>
</p> -->
