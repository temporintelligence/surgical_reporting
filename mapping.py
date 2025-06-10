from enum import Enum
from typing import Dict


class Instrument(Enum):
    GRASPER = "grasper"
    BIPOLAR = "bipolar"
    HOOK = "hook"
    SCISSORS = "scissors"
    CLIPPER = "clipper"
    IRRIGATOR = "irrigator"
    SPECIMEN_BAG = "specimen_bag"
    NO_INSTRUMENT = "no_instrument"


class Verb(Enum):
    GRASP = "grasp"
    RETRACT = "retract"
    DISSECT = "dissect"
    COAGULATE = "coagulate"
    CLIP = "clip"
    CUT = "cut"
    ASPIRATE = "aspirate"
    IRRIGATE = "irrigate"
    PACK = "pack"
    NULL_VERB = "null_verb"


class Target(Enum):
    GALLBLADDER = "gallbladder"
    CYSTIC_PLATE = "cystic_plate"
    CYSTIC_DUCT = "cystic_duct"
    CYSTIC_ARTERY = "cystic_artery"
    CYSTIC_PEDICLE = "cystic_pedicle"
    BLOOD_VESSEL = "blood_vessel"
    FLUID = "fluid"
    ABDOMINAL_WALL_CAVITY = "abdominal_wall_cavity"
    LIVER = "liver"
    ADHESION = "adhesion"
    OMENTUM = "omentum"
    PERITONEUM = "peritoneum"
    GUT = "gut"
    SPECIMEN_BAG = "specimen_bag"
    NULL_TARGET = "null_target"


class Phase(Enum):
    PREPARATION = "preparation"
    CALOT_TRIANGLE_DISSECTION = "calot-triangle-dissection"
    CLIPPING_AND_CUTTING = "clipping-and-cutting"
    GALLBLADDER_DISSECTION = "gallbladder-dissection"
    GALLBLADDER_PACKAGING = "gallbladder-packaging"
    CLEANING_AND_COAGULATION = "cleaning-and-coagulation"
    GALLBLADDER_EXTRACTION = "gallbladder-extraction"


class SurgeryMappings:
    instrument_mapping: Dict[int, str] = {i: inst.value for i, inst in enumerate(Instrument)}
    verb_mapping: Dict[int, str] = {i: verb.value for i, verb in enumerate(Verb)}
    target_mapping: Dict[int, str] = {i: target.value for i, target in enumerate(Target)}
    phase_mapping: Dict[int, str] = {i: phase.value for i, phase in enumerate(Phase)}

    # Flattened mapping for all detectable objects (instrument + target)
    # TODO: Need to make sure why 'specimen_bag' appear twice, and if this is not a problem.
    mapping: Dict[int, str] = {
        0: "grasper", 1: "bipolar", 2: "hook", 3: "scissors", 4: "clipper", 5: "irrigator", 6: "specimen_bag",
        7: "gallbladder", 8: "cystic_plate", 9: "cystic_duct", 10: "cystic_artery", 11: "cystic_pedicle",
        12: "blood_vessel", 13: "fluid", 14: "abdominal_wall_cavity", 15: "liver", 16: "adhesion",
        17: "omentum", 18: "peritoneum", 19: "gut", 20: "specimen_bag", 21: "null_target"
    }

    # Automatically generated reverse mapping
    reverse_mapping: Dict[str, int] = {v: k for k, v in mapping.items()}

