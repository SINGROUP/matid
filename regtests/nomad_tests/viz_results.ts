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
//let file = "./exciting2/Class2D/C4B2F4N2.json"                // Y: Too big 2D material, actually two 2D-materials together
//let file = "./exciting2/Class2D/C12H10N2.json"                // Y: The cell is too big, actually a disconnected 2D network of molecules

// Material2D
//let file = "./exciting2/Material2D/Adsorbate/C62H10N2.json"   // Y/Y
//let file = "./exciting2/Material2D/Pristine/B2N2.json"        // Y/Y
//let file = "./exciting2/Material2D/Pristine/BN.json"        // Y/Y
//let file = "./exciting2/Material2D/Pristine/C2.json"        // Y/Y
//let file = "./exciting2/Material2D/Pristine/C4F4.json"        // Y/Y
//let file = "./exciting2/Material2D/Pristine/C50.json"        // Y/Y
//let file = "./exciting2/Material2D/Substitution/C49N.json"        // Y/Y
//let file = "./exciting2/Material2D/Vacancy/C49.json"        // Y/Y

// Surface
//let file = "./exciting2/Surface/Pristine/W3.json"        // Y/Y

/******************************************************************************/
// FHIAIMS

// Class2D
//let file = "./fhiaims5/Class2D/Ba16Ge20O56.json"  // N: Amorphous surface
//let file = "./fhiaims5/Class2D/Ba16O40Si12.json"  // N: Surface, complicated
//let file = "./fhiaims5/Class2D/Ca32O80Zr24.json"  // N: Surface, but only one layer
//let file = "./fhiaims5/Class2D/F2.json"           // Y: Too sparse for a 2D material
//let file = "./fhiaims5/Class2D/Ge12Mg12O36.json"  // N: Surface, complicated
//let file = "./fhiaims5/Class2D/Mg12O36Ti12.json"  // N: Surface, complicated
//let file = "./fhiaims5/Class2D/Ne2.json"  // Y: Too sparse for a 2D material

// Material2D
//let file = "./fhiaims5/Material2D/Pristine/C2.json"  // Y/Y
//let file = "./fhiaims5/Material2D/Pristine/C72.json"  // Y/Y
//let file = "./fhiaims5/Material2D/Pristine/C128.json"  // Y/Y
//let file = "./fhiaims5/Material2D/Pristine/C338.json"  // Y/Y

// Surface
//let file = "./fhiaims5/Surface/Pristine/Ba16O48Si16.json" // Y/Y

//let file = "./fhiaims5/Surface/Adsorbate/C2Ba16Ge20O60.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C2Ba16O44Zr12.json" // Y/N
//let file = "./fhiaims5/Surface/Adsorbate/C2Ca32Ge24O84.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C2Ge16O48Sr12.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C2H2Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C2H3Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C2H4Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C2H5Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C2H6Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C2H6Ru42.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C2H7Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C2H8Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C2H9Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C2O48Sn16Sr12.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C2O70Si24Sr18.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C2O92Sr24Zr32.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C3H3Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C3H10Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C3H11Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C4H4Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C4Mg16O16.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C4Mg32O32.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C4Mg48O48.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C6H6Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C6H8Ru33.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C6H8Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C6H9Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C7H10Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C12H4O5Si14.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C18H3Si17.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C18H3Si18.json" // Y/N
//let file = "./fhiaims5/Surface/Adsorbate/C18H3Si19.json" // Y/N
//let file = "./fhiaims5/Surface/Adsorbate/C18H4Si17.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C18H4Si18.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C26H3Si18.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C26H6Si18.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C33Hf32O2.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C33Mo32O2.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C33Nb32O2.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C33O2Ta32.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C33O2Ti32.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C33O2V32.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C33O2Zr32.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C34H3Si18.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C34H4O5Si20.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C34H4Si18.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C48H16O20Si56.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C52H6Si36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C72H12O20Si80.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C104H12O20Si80.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C136H12O20Si80.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C150H16Si96.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C162H27O45Si180.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C234H27O45Si180.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/C306H27O45Si180.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/CBa12O46Ti16.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/CBa16O42Ti12.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/CGe16Mg12O46.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/CH4Mg61NiO62.json" // Y/N  Substitute vs adsorbate
//let file = "./fhiaims5/Surface/Adsorbate/CH6Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/CH7Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/CH8Ru36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/CHRu36.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/CMg24O74Si24.json" // Y/Y  If reconstructions are considered to be adsorbates
//let file = "./fhiaims5/Surface/Adsorbate/CMg39NiO40.json" // Y/Y  If reconstructions are considered to be adsorbates
//let file = "./fhiaims5/Surface/Adsorbate/CMg39NiO40.json" // Y/N  Substitution is not correct. Should refactor the code to use chemical environment to judge substitutions
//let file = "./fhiaims5/Surface/Adsorbate/CMg61NiO62.json" // Y/N  Substitute vs adsorbate
//let file = "./fhiaims5/Surface/Adsorbate/F2O9Sr3Ti3.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H2Mg39NiO40.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H2Mg48O49.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H2Mg55NiO56.json" // Y/N Substitute vs adsorbate
//let file = "./fhiaims5/Surface/Adsorbate/H2O25Sr24.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H2O49Sr48.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H4Mg16O16.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H4Mg48O50.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H4O50Sr48.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H6Mg48O51.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H8Mg48O52.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H8O52Sr48.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H10Mg24O29.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H10Mg48O53.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H10O53Sr48.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H12Mg48O48.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H12Mg48O54.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/H20Mg32O42.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/HAu16O.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/Mg24O72Si24.json" // Y/Y If reconstructions are considered to be adsorbates
//let file = "./fhiaims5/Surface/Adsorbate/Mg76NiO76.json" // Y/N Substitute vs adsorbate
//let file = "./fhiaims5/Surface/Adsorbate/Ne2O9Sr3Ti3.json" // Y/Y
//let file = "./fhiaims5/Surface/Adsorbate/NeO9Sr3Ti3.json" // Y/Y


//let file = "slanted1.json";
let filepath = dir + "/" + file
//let filepath2 = dir + "/nacl_basic.json"
let targetElement = document.getElementById("visualizationCanvas")
var viewer = new StructureViewer(targetElement, false, {showParam: false, showCopies: false, showTags: true, allowRepeat: false, showCell: true});
//var viewer = new StructureViewer(targetElement);
loadJSON(filepath, (json) => {viewer.load(json);}, loadError);
