# Trait-wise Wavelength Interpretation

This note maps BeamSearch K=30 selected wavelengths to broad vegetation spectral regions. The regions are interpretive aids, not proof of causal absorption.

## cab

cab uses 16 visible/red-edge bands, 10 NIR/water-adjacent bands, and 4 SWIR bands. Top regions: nir_swir_structure_water:6; red_edge_chlorophyll_structure:6; swir_water_1450_region:4; nir_leaf_structure:4. Expected visible, red, and red-edge dominance because chlorophyll absorbs strongly in blue/red and affects the red-edge.

- **blue_visible_pigment**: 424 nm. blue visible pigment absorption; chlorophyll/carotenoid sensitivity
- **green_visible_pigment**: 547, 548 nm. green visible reflectance; pigment balance and leaf color response
- **nir_leaf_structure**: 817, 841, 882, 954 nm. near-infrared scattering; leaf/canopy internal structure
- **nir_swir_structure_water**: 1129, 1170, 1188, 1193, 1198, 1264 nm. NIR/SWIR transition; structure and water-sensitive continuum
- **red_chlorophyll_absorption**: 637, 641, 649, 683 nm. red chlorophyll absorption zone
- **red_edge_chlorophyll_structure**: 697, 698, 710, 715, 716, 739 nm. red-edge transition; chlorophyll and canopy/leaf structure
- **swir_water_1450_region**: 1350, 1444, 1459, 1487 nm. around 1450 nm water absorption region
- **yellow_orange_visible**: 612, 616, 618 nm. yellow/orange visible transition; pigment shoulder region

## cw

cw uses 2 visible/red-edge bands, 6 NIR/water-adjacent bands, and 22 SWIR bands. Top regions: swir_dry_matter_protein_carbon:21; nir_swir_structure_water:4; green_visible_pigment:1; yellow_orange_visible:1. Expected water-sensitive NIR/SWIR regions near 970, 1450, and 1900 nm plus structural context.

- **green_visible_pigment**: 539 nm. green visible reflectance; pigment balance and leaf color response
- **nir_leaf_structure**: 937 nm. near-infrared scattering; leaf/canopy internal structure
- **nir_swir_structure_water**: 1106, 1184, 1289, 1297 nm. NIR/SWIR transition; structure and water-sensitive continuum
- **swir_dry_matter_protein_carbon**: 1545, 1591, 1592, 1593, 1596, 1597, 1601, 1602, 1604, 1606, 1607, 1608, 1609, 1642, 1646, 1648, 1652, 1664, 1668, 1697, 1705 nm. SWIR dry matter, protein, carbon and cellulose/lignin sensitivity
- **swir_water_1450_region**: 1434 nm. around 1450 nm water absorption region
- **water_970_nm_region**: 980 nm. near 970 nm water absorption feature
- **yellow_orange_visible**: 585 nm. yellow/orange visible transition; pigment shoulder region

## cm

cm uses 4 visible/red-edge bands, 4 NIR/water-adjacent bands, and 22 SWIR bands. Top regions: swir_dry_matter_protein_carbon:16; swir_protein_cellulose_lignin:4; water_970_nm_region:3; swir_water_1450_region:2. Expected SWIR dry matter and biochemical regions, especially 1500-1800 and 2000-2200 nm.

- **green_visible_pigment**: 510, 514 nm. green visible reflectance; pigment balance and leaf color response
- **nir_swir_structure_water**: 1081 nm. NIR/SWIR transition; structure and water-sensitive continuum
- **red_chlorophyll_absorption**: 660 nm. red chlorophyll absorption zone
- **red_edge_chlorophyll_structure**: 699 nm. red-edge transition; chlorophyll and canopy/leaf structure
- **swir_dry_matter_protein_carbon**: 1559, 1574, 1639, 1645, 1647, 1667, 1670, 1672, 1676, 1693, 1715, 1728, 1734, 1738, 1739, 1740 nm. SWIR dry matter, protein, carbon and cellulose/lignin sensitivity
- **swir_protein_cellulose_lignin**: 2157, 2166, 2168, 2176 nm. SWIR biochemical absorptions; protein, cellulose, lignin, starch
- **swir_water_1450_region**: 1300, 1332 nm. around 1450 nm water absorption region
- **water_970_nm_region**: 982, 987, 992 nm. near 970 nm water absorption feature

## LAI

LAI uses 14 visible/red-edge bands, 5 NIR/water-adjacent bands, and 11 SWIR bands. Top regions: red_chlorophyll_absorption:11; swir_protein_cellulose_lignin:7; nir_swir_structure_water:3; swir_dry_matter_protein_carbon:2. Expected NIR/red-edge structure and water/biomass covariates rather than a single pigment-only region.

- **green_visible_pigment**: 531 nm. green visible reflectance; pigment balance and leaf color response
- **nir_leaf_structure**: 746 nm. near-infrared scattering; leaf/canopy internal structure
- **nir_swir_structure_water**: 1054, 1066, 1222 nm. NIR/SWIR transition; structure and water-sensitive continuum
- **red_chlorophyll_absorption**: 649, 650, 651, 653, 654, 661, 662, 663, 664, 666, 681 nm. red chlorophyll absorption zone
- **red_edge_chlorophyll_structure**: 705 nm. red-edge transition; chlorophyll and canopy/leaf structure
- **swir_dry_matter_protein_carbon**: 1601, 1730 nm. SWIR dry matter, protein, carbon and cellulose/lignin sensitivity
- **swir_protein_cellulose_lignin**: 2055, 2057, 2058, 2059, 2060, 2061, 2085 nm. SWIR biochemical absorptions; protein, cellulose, lignin, starch
- **swir_water_1450_region**: 1324, 1340 nm. around 1450 nm water absorption region
- **water_970_nm_region**: 990 nm. near 970 nm water absorption feature
- **yellow_orange_visible**: 606 nm. yellow/orange visible transition; pigment shoulder region

## cp

cp uses 17 visible/red-edge bands, 6 NIR/water-adjacent bands, and 7 SWIR bands. Top regions: green_visible_pigment:7; swir_dry_matter_protein_carbon:7; red_chlorophyll_absorption:6; nir_swir_structure_water:5. Expected SWIR protein/nitrogen-related regions, with useful visible/NIR structural covariates.

- **blue_visible_pigment**: 485 nm. blue visible pigment absorption; chlorophyll/carotenoid sensitivity
- **green_visible_pigment**: 510, 519, 522, 533, 544, 551, 556 nm. green visible reflectance; pigment balance and leaf color response
- **nir_leaf_structure**: 793 nm. near-infrared scattering; leaf/canopy internal structure
- **nir_swir_structure_water**: 1041, 1211, 1226, 1241, 1275 nm. NIR/SWIR transition; structure and water-sensitive continuum
- **red_chlorophyll_absorption**: 630, 656, 664, 669, 675, 679 nm. red chlorophyll absorption zone
- **red_edge_chlorophyll_structure**: 721 nm. red-edge transition; chlorophyll and canopy/leaf structure
- **swir_dry_matter_protein_carbon**: 1542, 1573, 1665, 1736, 1767, 1768, 1770 nm. SWIR dry matter, protein, carbon and cellulose/lignin sensitivity
- **yellow_orange_visible**: 616, 623 nm. yellow/orange visible transition; pigment shoulder region

## cbc

cbc uses 7 visible/red-edge bands, 5 NIR/water-adjacent bands, and 18 SWIR bands. Top regions: swir_dry_matter_protein_carbon:14; nir_swir_structure_water:5; swir_water_1450_region:2; red_chlorophyll_absorption:2. Expected SWIR dry matter/carbon regions and strong similarity to Cm-related selections.

- **blue_visible_pigment**: 487 nm. blue visible pigment absorption; chlorophyll/carotenoid sensitivity
- **green_visible_pigment**: 518, 526 nm. green visible reflectance; pigment balance and leaf color response
- **nir_swir_structure_water**: 1007, 1090, 1104, 1264, 1271 nm. NIR/SWIR transition; structure and water-sensitive continuum
- **red_chlorophyll_absorption**: 646, 669 nm. red chlorophyll absorption zone
- **red_edge_chlorophyll_structure**: 709 nm. red-edge transition; chlorophyll and canopy/leaf structure
- **swir_dry_matter_protein_carbon**: 1536, 1581, 1588, 1596, 1603, 1618, 1662, 1685, 1687, 1706, 1737, 1795, 1796, 1797 nm. SWIR dry matter, protein, carbon and cellulose/lignin sensitivity
- **swir_protein_cellulose_lignin**: 2099, 2123 nm. SWIR biochemical absorptions; protein, cellulose, lignin, starch
- **swir_water_1450_region**: 1331, 1443 nm. around 1450 nm water absorption region
- **yellow_orange_visible**: 572 nm. yellow/orange visible transition; pigment shoulder region

## car

car uses 18 visible/red-edge bands, 6 NIR/water-adjacent bands, and 6 SWIR bands. Top regions: blue_visible_pigment:9; swir_water_1450_region:6; nir_swir_structure_water:5; red_chlorophyll_absorption:4. Expected visible blue-green and red-edge associations because carotenoids overlap pigment absorption zones.

- **blue_visible_pigment**: 400, 402, 439, 440, 441, 446, 457, 486, 487 nm. blue visible pigment absorption; chlorophyll/carotenoid sensitivity
- **green_visible_pigment**: 568 nm. green visible reflectance; pigment balance and leaf color response
- **nir_leaf_structure**: 909 nm. near-infrared scattering; leaf/canopy internal structure
- **nir_swir_structure_water**: 1033, 1078, 1225, 1275, 1291 nm. NIR/SWIR transition; structure and water-sensitive continuum
- **red_chlorophyll_absorption**: 675, 677, 679, 680 nm. red chlorophyll absorption zone
- **red_edge_chlorophyll_structure**: 704 nm. red-edge transition; chlorophyll and canopy/leaf structure
- **swir_water_1450_region**: 1431, 1455, 1459, 1472, 1474, 1475 nm. around 1450 nm water absorption region
- **yellow_orange_visible**: 596, 600, 607 nm. yellow/orange visible transition; pigment shoulder region

## anth

anth uses 22 visible/red-edge bands, 4 NIR/water-adjacent bands, and 4 SWIR bands. Top regions: green_visible_pigment:13; red_chlorophyll_absorption:4; swir_water_1450_region:4; red_edge_chlorophyll_structure:3. Expected visible pigment and red-edge associations, but interpretation is limited by sparse labels.

- **blue_visible_pigment**: 486 nm. blue visible pigment absorption; chlorophyll/carotenoid sensitivity
- **green_visible_pigment**: 512, 525, 526, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561 nm. green visible reflectance; pigment balance and leaf color response
- **nir_leaf_structure**: 767 nm. near-infrared scattering; leaf/canopy internal structure
- **nir_swir_structure_water**: 1138, 1297 nm. NIR/SWIR transition; structure and water-sensitive continuum
- **red_chlorophyll_absorption**: 643, 655, 675, 676 nm. red chlorophyll absorption zone
- **red_edge_chlorophyll_structure**: 694, 695, 721 nm. red-edge transition; chlorophyll and canopy/leaf structure
- **swir_water_1450_region**: 1348, 1463, 1470, 1471 nm. around 1450 nm water absorption region
- **water_970_nm_region**: 972 nm. near 970 nm water absorption feature
- **yellow_orange_visible**: 592 nm. yellow/orange visible transition; pigment shoulder region

