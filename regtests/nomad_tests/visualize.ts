import { StructureViewer } from "./structureviewer"
import { loadJSON, loadError, loadMaterial } from "./loadutils"


//============================================================================
// Visualize single system
let dir = "data/";

/******************************************************************************/
// Exciting
// Class2D
//let file = "./exciting5/Class2D/C12H10N2.json"        // N: Too sparse

// Material2D
//let file = "./exciting5/Material2D/C4F4+B2N2.json"    // Y/Y
//let file = "./exciting5/Material2D/C50+C12H10N2.json" // Y/Y
//let file = "./exciting5/Material2D/B2N2.json"         // Y/Y
//let file = "./exciting5/Material2D/BN.json"           // Y/Y
//let file = "./exciting5/Material2D/C2.json"           // Y/Y
//let file = "./exciting5/Material2D/C4F4.json"         // Y/Y
//let file = "./exciting5/Material2D/C50.json"          // Y/Y
//let file = "./exciting5/Material2D/C49+N.json"        // Y/Y
//let file = "./exciting5/Material2D/C49.json"          // Y/Y

// Surface
//let file = "./exciting5/Surface/W3.json"              // Y/Y

/******************************************************************************/
// FHIAIMS

// Class2D
//let file = "./fhiaims12/Class2D/Ba16Ge20O56.json"  // N: Amorphous
//let file = "./fhiaims12/Class2D/F2.json"           // N: Too sparse
//let file = "./fhiaims12/Class2D/Ne2.json"          // N: Too sparse

// Material2D
//let file = "./fhiaims12/Material2D/C2.json"    // Y/Y
//let file = "./fhiaims12/Material2D/C72.json"   // Y/Y
//let file = "./fhiaims12/Material2D/C128.json"  // Y/Y
//let file = "./fhiaims12/Material2D/C338.json"  // Y/Y

// Surface
//let file = "./fhiaims12/Surface/Au16+HO.json"         // Y/Y
//let file = "./fhiaims12/Surface/Ba12O44Ti16.json"     // Y/Y
//let file = "./fhiaims12/Surface/Ba12O44Ti16+CO2.json" // Y/Y
//let file = "./fhiaims12/Surface/Ba16+O40Si12.json"    // Y/N: Wrong cell is detected
//let file = "./fhiaims12/Surface/Ba16Ge20O56+C2O4.json"// Y/Y
//let file = "./fhiaims12/Surface/Ba16O40Ti12.json"     // Y/Y
//let file = "./fhiaims12/Surface/Ba16O40Ti12+CO2.json" // Y/Y
//let file = "./fhiaims12/Surface/Ba16O40Zr12+C2O4.json"// Y/Y
//let file = "./fhiaims12/Surface/Ba16O48Si16.json"     // Y/Y
//let file = "./fhiaims12/Surface/Ba20O52Ti20.json"     // Y/Y
//let file = "./fhiaims12/Surface/C8Mo16.json"          // Y/Y
//let file = "./fhiaims12/Surface/C8Mo16+CO2.json"      // Y/Y
//let file = "./fhiaims12/Surface/C9Si12+C9H3Si6.json"  // Y/Y
//let file = "./fhiaims12/Surface/C12Si12+H4O5Si2.json" // Y/Y
//let file = "./fhiaims12/Surface/C12Si15+C6H3Si4.json" // Y/Y
//let file = "./fhiaims12/Surface/C18Si17+H3.json"      // Y/Y
//let file = "./fhiaims12/Surface/C18Si17+H4.json"      // Y/Y
//let file = "./fhiaims12/Surface/C18Si18+C8H3.json"    // Y/Y
//let file = "./fhiaims12/Surface/C18Si18+C8H6.json"    // Y/Y
//let file = "./fhiaims12/Surface/C18Si18+C16H3.json"   // Y/Y
//let file = "./fhiaims12/Surface/C18Si18+C16H4.json"   // Y/Y
//let file = "./fhiaims12/Surface/C18Si18+C16H4O5Si2.json"  // Y/Y
//let file = "./fhiaims12/Surface/C18Si18+H3.json"  // Y/Y
//let file = "./fhiaims12/Surface/C18Si18+H4.json"  // Y/Y
//let file = "./fhiaims12/Surface/C32Hf32.json"  // Y/Y
//let file = "./fhiaims12/Surface/C32Hf32+CO2.json"  // Y/Y
//let file = "./fhiaims12/Surface/C32Mo32.json"  // Y/Y
//let file = "./fhiaims12/Surface/C32Mo32+CO2.json"  // Y/Y
//let file = "./fhiaims12/Surface/C32Nb32.json"  // Y/Y
//let file = "./fhiaims12/Surface/C32Nb32+CO2.json"  // Y/Y
//let file = "./fhiaims12/Surface/C32Ta32.json"  // Y/Y
//let file = "./fhiaims12/Surface/C32Ta32+CO2.json"  // Y/Y
//let file = "./fhiaims12/Surface/C32Ti32.json"  // Y/Y
//let file = "./fhiaims12/Surface/C32Ti32+CO2.json"  // Y/Y
//let file = "./fhiaims12/Surface/C32V32.json"  // Y/Y
//let file = "./fhiaims12/Surface/C32V32+CO2.json"  // Y/Y
//let file = "./fhiaims12/Surface/C32Zr32.json"  // Y/Y
//let file = "./fhiaims12/Surface/C32Zr32+CO2.json"  // Y/Y
//let file = "./fhiaims12/Surface/C36Si36+C16H6.json"  // Y/Y
//let file = "./fhiaims12/Surface/C48Si48+H16O20Si8.json"  // Y/Y
//let file = "./fhiaims12/Surface/C72Si72+C32H12O20Si8.json"  // Y/Y
//let file = "./fhiaims12/Surface/C72Si72+C64H12O20Si8.json"  // Y/Y
//let file = "./fhiaims12/Surface/C72Si72+H12O20Si8.json"  // Y/Y
//let file = "./fhiaims12/Surface/C96Si96+C54H16.json"  // Y/Y
//let file = "./fhiaims12/Surface/C162Si162+C72H27O45Si18.json"  // Y/Y
//let file = "./fhiaims12/Surface/C162Si162+C144H27O45Si18.json"  // Y/Y
//let file = "./fhiaims12/Surface/C162Si162+H27O45Si18.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ca8Ge6O20.json"        // Y/Y
//let file = "./fhiaims12/Surface/Ca12Ge16O44.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ca12O44Ti16.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ca16Ge12O40.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ca16O16+C.json"        // Y/Y
//let file = "./fhiaims12/Surface/Ca16O40Ti12.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ca32O80Zr24.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ca16O56Zr20.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ca24Ge32O88.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ca24O88Si32.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ca24O88Ti32.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ca24O88Zr32.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ca32Ge24O80.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ca32Ge24O80+C2O4.json" // Y/Y
//let file = "./fhiaims12/Surface/Ca32O80Ti24.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ge12Mg12O36.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ge12Mg16O40.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ge12O40Sr16.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ge16Mg12O44.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ge16Mg12O44+CO2.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ge16Mg16O48.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ge16O44Sr12.json"      // Y/Y
//let file = "./fhiaims12/Surface/Ge16O44Sr12+C2O4.json" // Y/Y
//let file = "./fhiaims12/Surface/Mg8O24Si8.json"        // Y/Y
//let file = "./fhiaims12/Surface/Mg10O30Si10.json"      // Y/Y
//let file = "./fhiaims12/Surface/Mg12O12+H4O2.json"     // Y/Y
//let file = "./fhiaims12/Surface/Mg12O36Si12.json"      // Y/Y
//let file = "./fhiaims12/Surface/Mg12O36Ti12.json"      // Y/Y
//let file = "./fhiaims12/Surface/Mg14O42Si14.json"      // Y/Y
//let file = "./fhiaims12/Surface/Mg16O16.json"          // Y/Y
//let file = "./fhiaims12/Surface/Mg16O16+C4.json"       // Y/Y
//let file = "./fhiaims12/Surface/Mg16O16+H4.json"       // Y/Y
//let file = "./fhiaims12/Surface/Mg16O48Si16.json"      // Y/Y
//let file = "./fhiaims12/Surface/Mg16O48Sn16.json"      // Y/Y
//let file = "./fhiaims12/Surface/Mg16O48Ti16.json"      // Y/Y
//let file = "./fhiaims12/Surface/Mg16O70Si24+CMg8O4.json"  // Y/Y There is reconstruction on the surface
//let file = "./fhiaims12/Surface/Mg16O71Si24+CMg8O3.json"  // Y/Y There is reconstruction on the surface
//let file = "./fhiaims12/Surface/Mg16O72Si24+Mg8.json"  // Y/Y There is reconstruction on the surface
//let file = "./fhiaims12/Surface/Mg20O68Si24.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg24O24.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg24O24+H10O5.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg32O32.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg32O32+C4.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg32O32+H20O10.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg39O40+CH4Ni.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg39O40+CNi.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg39O40+H2Ni.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg39O40+Ni.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg48O48.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg48O48+C4.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg48O48+H2O.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg48O48+H4O2.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg48O48+H6O3.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg48O48+H8O4.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg48O48+H10O5.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg48O48+H12.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg48O48+H12O6.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg55O56+CNi.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg55O56+H2Ni.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg61O62+H2Ni.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg61O62+CH4Ni.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg61O62+CNi.json"  // Y/Y
//let file = "./fhiaims12/Surface/Mg76O76+Ni.json"  // Y/Y
//let file = "./fhiaims12/Surface/O9Sr3Ti3.json"  // Y/Y
//let file = "./fhiaims12/Surface/O9Sr3Ti3+F2.json"  // Y/Y
//let file = "./fhiaims12/Surface/O9Sr3Ti3+Ne.json"  // Y/Y
//let file = "./fhiaims12/Surface/O9Sr3Ti3+Ne2.json"  // Y/Y
//let file = "./fhiaims12/Surface/O12Sr12.json"  // Y/Y
//let file = "./fhiaims12/Surface/O22Sr6Zr8.json"  // Y/Y
//let file = "./fhiaims12/Surface/O24Sr8Ti12.json"  // Y/Y
//let file = "./fhiaims12/Surface/O24Sr24+H2O.json"  // Y/Y
//let file = "./fhiaims12/Surface/O28Sr12Ti8.json"  // Y/Y
//let file = "./fhiaims12/Surface/O28Sr12Ti12.json"  // Y/Y
//let file = "./fhiaims12/Surface/O32Sr8Ti8.json"  // Y/Y
//let file = "./fhiaims12/Surface/O32Sr8Ti12.json"  // Y/Y
//let file = "./fhiaims12/Surface/O36Sr12Ti16.json"  // Y/Y
//let file = "./fhiaims12/Surface/O40Sn12Sr16.json"  // Y/Y
//let file = "./fhiaims12/Surface/O40Sr16Ti12.json"  // Y/Y
//let file = "./fhiaims12/Surface/O40Sr16Ti16.json"  // Y/Y
//let file = "./fhiaims12/Surface/O40Sr16Zr12.json"  // Y/Y
//let file = "./fhiaims12/Surface/O44Sn16Sr12.json"  // Y/Y
//let file = "./fhiaims12/Surface/O44Sn16Sr12+C2O4.json"  // Y/Y
//let file = "./fhiaims12/Surface/O44Sr12Ti12.json"  // Y/Y
//let file = "./fhiaims12/Surface/O44Sr12Ti16.json"  // Y/Y
//let file = "./fhiaims12/Surface/O44Sr12Zr16.json"  // Y/Y
//let file = "./fhiaims12/Surface/O48Sr16Ti16.json"  // Y/Y
//let file = "./fhiaims12/Surface/O48Sr48+H2O.json"  // Y/Y
//let file = "./fhiaims12/Surface/O48Sr48+H4O2.json"  // Y/Y
//let file = "./fhiaims12/Surface/O48Sr48+H8O4.json"  // Y/Y
//let file = "./fhiaims12/Surface/O48Sr48+H10O5.json"  // Y/Y
//let file = "./fhiaims12/Surface/O52Sr20Ti16.json"  // Y/Y
//let file = "./fhiaims12/Surface/O60Si18Sr24.json"  // Y/Y
//let file = "./fhiaims12/Surface/O66Si24Sr18.json"  // Y/Y
//let file = "./fhiaims12/Surface/O66Si24Sr18+C2O4.json"  // Y/Y
//let file = "./fhiaims12/Surface/O80Sr32Zr24.json"  // Y/Y
//let file = "./fhiaims12/Surface/O88Sr24Zr32.json"  // Y/Y
//let file = "./fhiaims12/Surface/O88Sr24Zr32+C2O4.json"  // Y/Y
//let file = "./fhiaims12/Surface/O96Sr32Ti32.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru33+C6H8.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C2H2.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C2H3.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C2H4.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C2H5.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C2H6.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C2H7.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C2H8.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C2H9.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C3H3.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C3H10.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C3H11.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C4H4.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C6H6.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C6H8.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C6H9.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+C7H10.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+CH.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+CH6.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+CH7.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru36+CH8.json"  // Y/Y
//let file = "./fhiaims12/Surface/Ru42+C2H6.json"  // Y/Y

//let file = "slanted1.json";
let filepath = dir + "/" + file
//let filepath2 = dir + "/nacl_basic.json"
let targetElement = document.getElementById("visualizationCanvas")
let options = {
    showParam: false,
    showCopies: false,
    showTags: true,
    allowRepeat: false,
    showCell: true,
    wrap: false,
    showLegend: false,
    showOptions: false
}
var viewer = new StructureViewer(targetElement, false, options);
//var viewer = new StructureViewer(targetElement);
loadJSON(filepath, (json) => {viewer.load(json);}, loadError);
