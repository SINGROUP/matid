import { StructureViewer } from "./structureviewer"
import { loadJSON, loadError, loadMaterial } from "./loadutils"


//============================================================================
// Visualize single system
let dir = "data/";
//let file = "P1t2Cu-BqFfSG4lERRdBoaOHezIR-.json";
//let file = "P0Aac7f72_7ocqf3mL1JC8Bok_p-o.json";
//let file = "graphene_3_viz.json";
//let file = "nacl_basic.json";
//let file = "nacl_stretched.json";
//let file = "nacl_dislocated.json";
//let file = "nacl_point_defect.json";
//let file = "bcc_bulk.json";
//let file = "bcc_111_surface.json";
//let file = "2d/C2EkVsG3bVq-fRguiJrFgQhb3nB7y.json";  // Defect graphene (vacancy)
//let file = "graphene.json";
//let file = "ewald.json";
//let file = "atom_ads.json";
//let file = "semimetal.json";
//let file = "1d.json";
//let file = "2d.json";
//let file = "2d_2.json";
//let file = "2d_3.json";
//let file = "2d_4.json";


/******************************************************************************/
// Exciting
// Class2D
//let file = "./exciting2/Class2D/C4B2F4N2.json"               // N: Too big 2D material, actually two 2D-materials together
//let file = "./exciting2/Class2D/C12H10N2.json"               // N: The cell is too big, actually a disconnected 2D network of molecules

// Material2D
//let file = "./exciting2/Material2D/Adsorbate/C62H10N2.json"  // Y/Y
//let file = "./exciting2/Material2D/Pristine/B2N2.json"       // Y/Y
//let file = "./exciting2/Material2D/Pristine/BN.json"         // Y/Y
//let file = "./exciting2/Material2D/Pristine/C2.json"         // Y/Y
//let file = "./exciting2/Material2D/Pristine/C4F4.json"       // Y/Y
//let file = "./exciting2/Material2D/Pristine/C50.json"        // Y/Y
//let file = "./exciting2/Material2D/Substitution/C49N.json"   // Y/Y
//let file = "./exciting2/Material2D/Vacancy/C49.json"         // Y/Y

// Surface
//let file = "./exciting2/Surface/Pristine/W3.json"            // Y/Y

/******************************************************************************/
// FHIAIMS

// Class2D
//let file = "./fhiaims6/Class2D/Ba16Ge20O56.json"  // N: Amorphous surface
//let file = "./fhiaims6/Class2D/Ba16O40Si12.json"  // N: Surface, but only one layer
//let file = "./fhiaims6/Class2D/Ca32O80Zr24.json"  // N: Surface, but only one layer
//let file = "./fhiaims6/Class2D/F2.json"           // Y: Too sparse for a 2D material
//let file = "./fhiaims6/Class2D/Ge12Mg12O36.json"  // N: Surface, the position tolerance is too big.
//let file = "./fhiaims6/Class2D/Mg12O36Ti12.json"  // N: Surface, the position tolerance is too big.
//let file = "./fhiaims6/Class2D/Ne2.json"          // Y: Too sparse for a 2D material

// Material2D
//let file = "./fhiaims6/Material2D/Pristine/C2.json"  // Y/Y
//let file = "./fhiaims6/Material2D/Pristine/C72.json"  // Y/Y
//let file = "./fhiaims6/Material2D/Pristine/C128.json"  // Y/Y
//let file = "./fhiaims6/Material2D/Pristine/C338.json"  // Y/Y

// Surface Pristine
//let file = "./fhiaims6/Surface/Pristine/Mg12O36Si12.json"  // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ba12O44Ti16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ba16O40Ti12.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ba16O48Si16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ba20O52Ti20.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/C8Mo16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/C32Hf32.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/C32Mo32.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/C32Nb32.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/C32Ta32.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/C32Ti32.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/C32V32.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/C32Zr32.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ca8Ge6O20.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ca12Ge16O44.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ca12O44Ti16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ca16Ge12O40.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ca16O40Ti12.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ca16O56Zr20.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ca24Ge32O88.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ca24O88Si32.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ca24O88Ti32.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ca24O88Zr32.json" // Y/Y Bug in detecting atoms in the unit cell. Does not interfere with surface search.
//let file = "./fhiaims6/Surface/Pristine/Ca32Ge24O80.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ge12Mg16O40.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ge12O40Sr16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ge16Mg12O44.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ge16Mg16O48.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Ge16O44Sr12.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Mg8O24Si8.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Mg10O30Si10.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Mg12O36Si12.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Mg14O42Si14.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Mg16O16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Mg16O48Si16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Mg16O48Sn16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Mg16O48Ti16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Mg24O24.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Mg32O32.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/Mg48O48.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O9Sr3Ti3.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O12Sr12.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O22Sr6Zr8.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O24Sr8Ti12.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O28Sr12Ti8.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O28Sr12Ti12.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O32Sr8Ti8.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O32Sr8Ti12.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O36Sr12Ti16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O40Sn12Sr16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O40Sr16Ti12.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O40Sr16Ti16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O40Sr16Zr12.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O44Sn16Sr12.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O44Sr12Ti12.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O44Sr12Ti16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O44Sr12Zr16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O48Sr16Ti16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O52Sr20Ti16.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O60Si18Sr24.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O66Si24Sr18.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O88Sr24Zr32.json" // Y/Y
//let file = "./fhiaims6/Surface/Pristine/O96Sr32Ti32.json" // Y/Y

// Surface + Adsorbates
//let file = "./fhiaims6/Surface/Adsorbate/C2Ba16Ge20O60.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C2Ba16O44Zr12.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C2Ca32Ge24O84.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C2Ge16O48Sr12.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C2H2Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C2H3Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C2H4Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C2H5Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C2H6Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C2H6Ru42.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C2H7Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C2H8Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C2H9Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C2O48Sn16Sr12.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C2O70Si24Sr18.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C2O92Sr24Zr32.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C3H3Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C3H10Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C3H11Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C4H4Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C4Mg16O16.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C4Mg32O32.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C4Mg48O48.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C6H6Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C6H8Ru33.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C6H8Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C6H9Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C7H10Ru36.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C12H4O5Si14.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C18H3Si17.json"   // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C18H3Si18.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C18H3Si19.json" // Y/Y If reconstructions are considered to be adsorbates
//let file = "./fhiaims6/Surface/Adsorbate/C18H4Si17.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C18H4Si18.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C26H3Si18.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C26H6Si18.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C33Hf32O2.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C33Mo32O2.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C33Nb32O2.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C33O2Ta32.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C33O2Ti32.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C33O2V32.json"  // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C33O2Zr32.json"       // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C34H3Si18.json"       // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C34H4O5Si20.json"     // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C34H4Si18.json"       // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C48H16O20Si56.json"   // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C52H6Si36.json"       // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C72H12O20Si80.json"   // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C104H12O20Si80.json"  // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C136H12O20Si80.json"  // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C150H16Si96.json"     // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C162H27O45Si180.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C234H27O45Si180.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/C306H27O45Si180.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/CBa12O46Ti16.json"    // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/CBa16O42Ti12.json"// Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/CCa16O16.json"    // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/CGe16Mg12O46.json"// Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/CH6Ru36.json"     // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/CH7Ru36.json"     // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/CH8Ru36.json"     // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/CHRu36.json"      // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/CMg24O74Si24.json"// Y/Y If reconstructions are considered to be adsorbates
//let file = "./fhiaims6/Surface/Adsorbate/H2Mg39NiO40.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H2Mg48O49.json"   // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H2O25Sr24.json"   // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H2O49Sr48.json"   // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H4Mg12O14.json"   // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H4Mg16O16.json"   // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H4Mg48O50.json"   // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H4O50Sr48.json"   // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H6Mg48O51.json"   // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H8Mg48O52.json"   // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H8O52Sr48.json"   // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H10Mg24O29.json"  // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H10Mg48O53.json"  // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H10O53Sr48.json"  // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H12Mg48O48.json"  // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H12Mg48O54.json"  // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/H20Mg32O42.json"  // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/HAu16O.json"      // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/Mg24O72Si24.json" // Y/Y If reconstructions are considered to be adsorbates
//let file = "./fhiaims6/Surface/Adsorbate/Ne2O9Sr3Ti3.json" // Y/Y
//let file = "./fhiaims6/Surface/Adsorbate/NeO9Sr3Ti3.json"  // Y/Y

// Surface + Interstitial + Adsorbate
//let file = "./fhiaims6/Surface/Interstitial+Adsorbate/C8Mo16.json"       // N/N: The wrong cell is identified. Cannot be properly fixed because there is not enough information available in the system.
//let file = "./fhiaims6/Surface/Interstitial+Adsorbate/C9Mo16O2.json"     // N/N: The wrong cell is identified. Cannot be properly fixed because there is not enough information available in the system.
//let file = "./fhiaims6/Surface/Interstitial+Adsorbate/Ca32O80Ti24.json"  // Y/N: The whole surface is not detected. Increasing pos_tol fixes the problem.
//let file = "./fhiaims6/Surface/Interstitial+Adsorbate/CMg24O74Si24.json" // Y/N: The whole surfae is not detected. Increasing pos_tol detects the adorbates and reconstructed atoms nicely.
//let file = "./fhiaims6/Surface/Interstitial+Adsorbate/O80Sr32Zr24.json"  // Y/N: The whole surface is not detected. Increasing pos_tol fixes the problem.

// Surface + Substitution + Adsorbate
//let file = "./fhiaims6/Surface/Substitution+Adsorbate/CH4Mg39NiO40.json"  // Y/Y
//let file = "./fhiaims6/Surface/Substitution+Adsorbate/CH4Mg61NiO62.json"  // Y/Y
//let file = "./fhiaims6/Surface/Substitution+Adsorbate/CMg39NiO40.json"    // Y/Y
//let file = "./fhiaims6/Surface/Substitution+Adsorbate/CMg55NiO56.json"    // Y/Y
//let file = "./fhiaims6/Surface/Substitution+Adsorbate/CMg61NiO62.json"    // Y/Y
//let file = "./fhiaims6/Surface/Substitution+Adsorbate/H2Mg55NiO56.json"   // Y/Y

// Surface + Substitution
//let file = "./fhiaims6/Surface/Substitution/F2O9Sr3Ti3.json"  // Y/Y
//let file = "./fhiaims6/Surface/Substitution/Mg39NiO40.json"   // Y/Y
//let file = "./fhiaims6/Surface/Substitution/Mg76NiO76.json"   // Y/Y

// Surface + Vacancy + Adsorbate
//let file = "./fhiaims6/Surface/Vacancy+Adsorbate/Mg20O68Si24.json"  // Y/N The whole surface is not detected. Fixed by increasing pos_tol

// Vacancy + Interstitional + Substitution + Adsorbate
//let file = "./fhiaims6/Surface/Vacancy+Interstitial+Substitution+Adsorbate/H2Mg61NiO62.json"  // Y/N The whole surface is not detected. Fixed by increasing pos_tol.

//let file = "slanted1.json";
let filepath = dir + "/" + file
//let filepath2 = dir + "/nacl_basic.json"
let targetElement = document.getElementById("visualizationCanvas")
var viewer = new StructureViewer(targetElement, false, {showParam: false, showCopies: false, showTags: true, allowRepeat: false, showCell: true});
//var viewer = new StructureViewer(targetElement);
loadJSON(filepath, (json) => {viewer.load(json);}, loadError);
