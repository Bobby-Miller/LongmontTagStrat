from dataclasses import dataclass
from typing import Dict, Literal, Set
import pathlib
import pandas as pd
import logging
import json

from pandas.core.dtypes.dtypes import np
from tag_strats import (
    ATOMIC_STRAT_DICT,
    AtomicStrategy,
    Binding,
    TagSample,
    TYPE_OPTIONS_LIST,
    AtomicTag,
    AtomicData,
    TypeOptions,
    binding,
)


path = str(pathlib.Path(__file__).parent.resolve()) + "\\"
file_io_path = path + "file_io\\"
file_log_path = path + "file_log\\"
# Logging Path
log_file = file_log_path + "process logs 6_20.txt"
logger = logging.getLogger(__name__)
# xl_file_path = 'Wes Lancaster Type Definitions.XLSX'
### Inputs
xl_file_path = file_io_path + "longmont_cross_reference_062425.xlsx"
instance_sheet_name = file_io_path + "longmont_cross_reference_062425"
# Built from ignition script
### Outputs
opc_json_file_name = file_io_path + "tag_dict.json"
alt_json_file_name = file_io_path + "alt_tag_dict.json"
strategy_book_file_output = file_io_path + "Longmont Template Strategy out 6_23_B.xlsx"
strategy_book_file_input = file_io_path + "Longmont Template Strategy Merge 6_23_B.xlsx"
udt_file_output = file_io_path + "udts_6_24.json"
instance_file_output = file_io_path + "instances_6_24.json"
json_file_name = file_io_path + "longmont_base_udt_atomics.json"

logging.basicConfig(filename=log_file, level=logging.DEBUG)

type BaseUDTName = Literal[
    "ALLEN_BRADLEY/DIGITAL",
    "ALLEN_BRADLEY/DIGITAL_SINGLE_TAG",
    "ALLEN_BRADLEY/INTERLOCK",
    "ALLEN_BRADLEY/MOTOR",
    "ALLEN_BRADLEY/PID",
    "ALLEN_BRADLEY/VALVE",
    "ALLEN_BRADLEY/ANALOG",
    "ALLEN_BRADLEY/ANALOG_SINGLE_TAG",
]
BASE_UDT_NAME = [
    "ALLEN_BRADLEY/DIGITAL",
    "ALLEN_BRADLEY/DIGITAL_SINGLE_TAG",
    "ALLEN_BRADLEY/INTERLOCK",
    "ALLEN_BRADLEY/MOTOR",
    "ALLEN_BRADLEY/PID",
    "ALLEN_BRADLEY/VALVE",
    "ALLEN_BRADLEY/ANALOG",
    "ALLEN_BRADLEY/ANALOG_SINGLE_TAG",
]

type ExtendedUDTName = str
type AtomicName = str
type OPCTemplate = str
type ExtendedUDTDict = dict[ExtendedUDTName, BaseUDTName]


def build_extended_udt_map(
    xl_file_path: str,
    instance_sheet_name: str,
) -> ExtendedUDTDict:
    df = pd.read_excel(xl_file_path, sheet_name=instance_sheet_name)
    udt_dict = dict(zip(df["extended_udt"], df["base_udt"]))
    filtered_udt_dict = {}
    for key, val in udt_dict.items():
        if not isinstance(val, str):
            continue
        new_val = val.replace("\\", "/").replace(" _", "_")
        filtered_udt_dict[key] = new_val
    print("Extended Map Built")
    return filtered_udt_dict


type BaseUDTAtomics = dict[BaseUDTName, list[str]]


def base_tag_dict_import() -> BaseUDTAtomics:
    """
    returns {base_udt: list_of_atomic_opc_items}
    """
    with open(json_file_name, "r") as file:
        json_dict = json.load(file)
        return json_dict


SampleDict = Dict[str, Dict[TagSample, str]]
AtomicDict = Dict[str, Set[AtomicTag]]
# DeploymentSampleDict = Dict[str, Dict[TagSample, AtomicDeployment]]


def generate_udt_atomic_samples(
    extended_udt_map: ExtendedUDTDict,
    base_tag_dict: BaseUDTAtomics,
    xl_file_path: str,
    instance_sheet_name: str,
) -> SampleDict:
    """
    Process to generate list of samples for strategy generation
    """
    # Sampling for strategy spread
    template_model_dict: SampleDict = {key: {} for key in extended_udt_map.keys()}
    print("generated template_model_dict")
    df = pd.read_excel(xl_file_path, sheet_name=instance_sheet_name)
    plc_tags = df["opcitempath"]
    plc_tag_types = df["ignition_datatype"]
    udt_defs = df["extended_udt"]
    ign_atoms = df["ignition_atomic_tag"]
    udt_tag_names = df["udt_instance"]
    for idx, plc_tag in enumerate(plc_tags):
        tag_type = plc_tag_types.iloc[idx]
        udt_def = udt_defs.iloc[idx]
        ign_atom = ign_atoms.iloc[idx]
        udt_tag_name = udt_tag_names.iloc[idx]

        if not isinstance(tag_type, str):
            continue
        if not isinstance(udt_def, str) and not isinstance(udt_def, int):
            continue
        if not isinstance(ign_atom, str):
            continue
        if not isinstance(plc_tag, str):
            continue
        if not isinstance(udt_tag_name, str) and not isinstance(udt_tag_name, int):
            continue

        plc_tag = plc_tag.replace("{topic}.", "")

        tag_type = "Float" if tag_type == "Double" else tag_type
        udt_tag_name = str(udt_tag_name)
        ign_atom = ign_atom.replace("\\", "/").replace("\n", "").strip()
        if "/" in ign_atom:
            tag_sample = TagSample(ign_atom, tag_type, "waste")
            try:
                if tag_sample not in template_model_dict[udt_def].keys():
                    template_model_dict[udt_def][tag_sample] = (
                        udt_tag_name + "|" + plc_tag
                    )
            except KeyError:
                logger.debug(
                    f"Line Skipped: {tag_type=} | {udt_def=} | {ign_atom=} | {plc_tag=} |> udt_def not found in Instances sheet"
                )
            continue
        try:
            base_udt = (
                extended_udt_map[udt_def]
                .replace("AllenBradley", "ALLEN_BRADLEY")
                .strip()
            )
        except KeyError:
            logger.debug(
                f"Line Skipped: {tag_type=} | {udt_def=} | {ign_atom=} | {plc_tag=} |> udt_def not found in Instances sheet"
            )
            continue
        base_atoms = base_tag_dict[base_udt]
        if ign_atom not in base_atoms:
            tag_sample = TagSample(ign_atom, tag_type, "extension")
            if tag_sample in template_model_dict[udt_def].keys():
                continue
            template_model_dict[udt_def][tag_sample] = udt_tag_name + "|" + plc_tag
            continue
        tag_sample = TagSample(ign_atom, tag_type, "opc")
        if tag_sample in template_model_dict[udt_def].keys():
            continue
        template_model_dict[udt_def][tag_sample] = udt_tag_name + "|" + plc_tag
    return template_model_dict


@dataclass
class Atomic:
    ign_name: str
    opc_path: str | Binding
    data_type: Literal["Float", "Double", "Integer", "Boolean", "String"]


@dataclass
class ExtendedUDT:
    udt_name: str
    base_type: str
    atomics: list[Atomic]
    parameters: list[str]


type StratClass = Literal["opc", "waste", "extension"]
type UDTData = dict[ExtendedUDTName, ExtendedUDT]


class StrategySheet:
    def __init__(self, strategy_xl, max_parameters):
        self.strat_sheet = pd.read_excel(strategy_xl, "Sheet1")
        self.strat_udts: list[ExtendedUDTName] = self.strat_sheet["Extended UDT"]  # type: ignore
        self.ign_atomic_names: list[str] = self.strat_sheet["Ignition Name"]  # type: ignore
        self.strat_ign_data_types: list[TypeOptions] = self.strat_sheet["Data Type"]  # type: ignore
        self.strat_classes: list[StratClass] = self.strat_sheet["Atomic Type"]  # type: ignore
        self.samples: list[str] = self.strat_sheet["Sample PLC Tag"]  # type: ignore
        self.ign_instances: list[str] = self.strat_sheet["Sample UDT Instance Name"]  # type: ignore
        self.template_strats: list[AtomicStrategyString] = self.strat_sheet[
            "Template Strategy"
        ]  # type: ignore
        self.overwrites: list[Literal["X", ""]] = self.strat_sheet["Overwrite Option"]  # type: ignore
        self.parameters: dict[int, list] = {  # type: ignore
            i: self.strat_sheet[f"Parameter {i}"]
            for i in range(1, max_parameters + 1)  # type: ignore
        }


def create_udt_data(
    strategy_sheet: StrategySheet,
    extended_udt_map: ExtendedUDTDict,
) -> UDTData:
    """
    Main function for analyzing lancaster sheet
    """
    ss = strategy_sheet
    udt_dict: dict[ExtendedUDTName, ExtendedUDT] = {}
    for idx, udt in enumerate(ss.strat_udts):
        if ss.strat_classes[idx] == "waste":
            continue
        if udt not in udt_dict.keys():
            udt_dict[udt] = ExtendedUDT(
                udt,
                extended_udt_map[udt].replace("AllenBradley", "ALLEN_BRADLEY").strip(),
                [],
                [],
            )
        Strat, param_count = ATOMIC_STRAT_DICT[ss.template_strats[idx]]

        params = [ss.parameters[i][idx] for i in range(1, param_count + 1)]
        for p_idx, param in enumerate(params):
            # coerce np numbers to int
            if isinstance(param, np.float64) and not np.isnan(param):
                params[p_idx] = int(param)
                continue
            # coerce nan to None
            if isinstance(param, np.float64):
                params[p_idx] = None
        sample = ss.samples[idx]
        ign_inst = ss.ign_instances[idx]
        ign_atomic_name = ss.ign_atomic_names[idx]
        overwrite = ss.overwrites[idx] if isinstance(ss.overwrites[idx], str) else ""
        strat: AtomicStrategy = Strat(  # type: ignore
            AtomicData(
                sample,
                ign_inst,
                ign_atomic_name,
            ),
            overwrite,
            *params,
        )
        udt_dict[udt].atomics.append(
            Atomic(
                ign_atomic_name,
                strat.template,
                ss.strat_ign_data_types[idx],
            )
        )
        for param in strat.template_parameters:
            if not param:
                continue
            if param in udt_dict[udt].parameters:
                continue
            udt_dict[udt].parameters.append(param)

    return udt_dict


def create_xl_strategy_book(
    output_name: str,
    sample_dict: SampleDict,
    number_parameters: int,
) -> None:
    data = {
        "Extended UDT": [],
        "Ignition Name": [],
        "Data Type": [],
        "Atomic Type": [],
        "Sample UDT Instance Name": [],
        "Sample PLC Tag": [],
        "Template Strategy": [],
    }
    for i in range(1, number_parameters + 1):
        data[f"Parameter {i}"] = []
    data["Overwrite Option"] = []

    for key, val in sample_dict.items():
        for sample, tag_info in val.items():
            udt_inst = tag_info.split("|")[0]
            sample_tag = tag_info.split("|")[1]
            data["Extended UDT"].append(key)
            data["Ignition Name"].append(sample.ign_name)
            data["Data Type"].append(sample.datatype)
            data["Atomic Type"].append(sample.atomic_type)
            data["Sample UDT Instance Name"].append(udt_inst)
            data["Sample PLC Tag"].append(sample_tag)
            data["Template Strategy"].append("")
            data["Overwrite Option"].append("")
            for i in range(1, number_parameters + 1):
                data[f"Parameter {i}"].append("")
    df = pd.DataFrame(data)
    df.to_excel(output_name, index=False, engine="openpyxl")


@dataclass
class TemplateStrategy:
    opc_template: OPCTemplate
    strat_class: StratClass
    template_strat: AtomicStrategy
    overwrite: bool
    parameters: dict[int, str | int]


type UDTAtomicTemplateMap = dict[tuple[ExtendedUDTName, AtomicName], TemplateStrategy]


def atomic_check_dict_build(
    udt_data: UDTData, ss: StrategySheet
) -> UDTAtomicTemplateMap:
    atomic_check_dict = {}
    for key, val in udt_data.items():
        for atomic in val.atomics:
            for idx, udt in enumerate(ss.strat_udts):
                if (udt, ss.ign_atomic_names[idx]) == (key, atomic.ign_name):
                    Strat, param_count = ATOMIC_STRAT_DICT[ss.template_strats[idx]]
                    params = [ss.parameters[i][idx] for i in range(1, param_count + 1)]
                    for p_idx, param in enumerate(params):
                        # coerce np numbers to int
                        if isinstance(param, np.float64) and not np.isnan(param):
                            params[p_idx] = int(param)
                            continue
                        # coerce nan to None
                        if isinstance(param, np.float64):
                            params[p_idx] = None
                    try:
                        atomic_check_dict[(key, atomic.ign_name)] = TemplateStrategy(
                            atomic.opc_path["binding"]
                            if isinstance(atomic.opc_path, dict)
                            else atomic.opc_path,  # type: ignore
                            ss.strat_classes[idx],
                            Strat,
                            ss.overwrites[idx],  # type: ignore
                            {idx + 1: param for idx, param in enumerate(params)},
                        )
                    except TypeError:
                        breakpoint()
    return atomic_check_dict


@dataclass
class UDTAtomicInstanceCheck:
    udt: ExtendedUDTName
    ign_atomic_name: AtomicName
    nonconforming_list: list
    checked: int = 0  # Atomic Instances checked in sheet
    conforming: int = 0  # Atomic Instances conforming to atomic pattern


def udt_atomic_instance_conformance_check(
    udt_atomic_map: UDTAtomicTemplateMap,
    xl_file_path: str,
    instance_sheet_name: str,
    max_non_conforming: int,
) -> list[UDTAtomicInstanceCheck]:
    conformance_dict = {key: [0, 0, []] for key in udt_atomic_map.keys()}

    ### New Approach to reading excel
    df = pd.read_excel(xl_file_path, sheet_name=instance_sheet_name)
    plc_tags = df["opcitempath"]
    plc_tag_types = df["ignition_datatype"]
    udt_defs = df["extended_udt"]
    ign_atoms = df["ignition_atomic_tag"]
    ign_tag_names = df["udt_instance"]
    for idx, plc_tag in enumerate(plc_tags):
        tag_type = plc_tag_types.iloc[idx]
        udt_def = udt_defs.iloc[idx]
        ign_atom = ign_atoms.iloc[idx]
        ign_tag_name = ign_tag_names.iloc[idx]

        if not isinstance(tag_type, str):
            continue
        if not isinstance(udt_def, str) and not isinstance(udt_def, int):
            continue
        if not isinstance(ign_atom, str):
            continue
        if not isinstance(plc_tag, str):
            continue
        if not isinstance(ign_tag_name, str) and not isinstance(ign_tag_name, int):
            continue
        ign_tag_name = str(ign_tag_name)
        plc_tag = plc_tag.replace("{topic}.", "")
        tag_type = "Float" if tag_type == "Double" else tag_type
        ign_tag_name = str(ign_tag_name)
        ign_atom = ign_atom.replace("\\", "/").replace("\n", "").strip()
        if "/" in ign_atom:
            logger.debug(
                f"CONFORMANCE: Line Skipped: {tag_type=} | {udt_def=} | {ign_atom=} | {plc_tag=} |> No tags permitted on properties"
            )
            continue
        if tag_type not in TYPE_OPTIONS_LIST:
            logger.debug(
                f"CONFORMANCE: Line Skipped: {tag_type=} | {udt_def=} | {ign_atom=} | {plc_tag=} |> tag type not in possible tag types"
            )
            continue
        try:
            template_strategy = udt_atomic_map[(udt_def, ign_atom)]
        except KeyError:
            logger.debug(
                f"CONFORMANCE: Line Skipped: {tag_type=} | {udt_def=} | {ign_atom=} | {plc_tag=} |> could not find atomic match from strategy template"
            )
            continue

        if template_strategy.strat_class == "waste":
            logger.debug(
                f"CONFORMANCE: Line Skipped: {tag_type=} | {udt_def=} | {ign_atom=} | {plc_tag=} |> strategy class sheet defines this combination as 'waste'"
            )
        Strat = template_strategy.template_strat
        params = template_strategy.parameters.values()

        conformance_dict[(udt_def, ign_atom)][0] += 1

        try:
            strat: AtomicStrategy = Strat(  # type: ignore
                AtomicData(
                    plc_tag,
                    ign_tag_name,  # type: ignore
                    ign_atom,
                ),
                False,
                *params,
            )
        except IndexError:
            conformance_dict[(udt_def, ign_atom)][1] += 1
            continue
        template = (
            strat.template["binding"]
            if isinstance(strat.template, dict)
            else strat.template
        )
        if template_strategy.opc_template == template:  # type: ignore
            conformance_dict[(udt_def, ign_atom)][1] += 1
        elif len(conformance_dict[(udt_def, ign_atom)][2]) < max_non_conforming:
            conformance_dict[(udt_def, ign_atom)][2].append(plc_tag)

    return [
        UDTAtomicInstanceCheck(
            key[0],
            key[1],
            val[2],
            val[0],
            val[1],
        )
        for key, val in conformance_dict.items()
    ]


def create_conformance_book(
    output_name: str,
    atomic_check_list: list[UDTAtomicInstanceCheck],
    num_nonconforming: int,
) -> None:
    data = {
        "Extended UDT": [],
        "Ignition Name": [],
        "Checked": [],
        "Conforming": [],
    }
    for i in range(1, num_nonconforming + 1):
        data[f"Nonconformance {i}"] = []

    for ac in atomic_check_list:
        data["Extended UDT"].append(ac.udt)
        data["Ignition Name"].append(ac.ign_atomic_name)
        data["Checked"].append(ac.checked)
        data["Conforming"].append(ac.conforming)
        for i in range(num_nonconforming):
            try:
                data[f"Nonconformance {i + 1}"].append(ac.nonconforming_list[i])
            except IndexError:
                data[f"Nonconformance {i + 1}"].append("")
    df = pd.DataFrame(data)
    df.to_excel(output_name, index=False, engine="openpyxl")


@dataclass
class TagInstance:
    parameters: dict[str, str | Binding]


@dataclass(frozen=True)
class InstanceKey:
    ign_inst_name: str
    udt_type: ExtendedUDTName
    base_path: str
    device: str


type InstData = dict[InstanceKey, TagInstance]


def tag_instance_build(
    udt_atomic_map: UDTAtomicTemplateMap,
    xl_file_path: str,
) -> InstData:
    instance_dict: InstData = {}

    ### New Approach
    df = pd.read_excel(xl_file_path)
    context = "INSTANCES"
    plc_tags = df["opcitempath"]
    plc_tag_types = df["ignition_datatype"]
    udt_defs = df["extended_udt"]
    ign_atoms = df["ignition_atomic_tag"]
    ign_tag_names = df["udt_instance"]
    plcs = df["device_id"]
    base_paths = df["ignition_folder"]
    descriptions = df["Description"]
    facility_ids = df["FacilityID"]
    ### Metadata Additions
    equip_types = df["EquipType"]
    facilities = df["Facility"]
    are_active = df["IsActive"]
    services = df["Service"]
    sites = df["Site"]
    dispositions = df["Disposition"]
    products = df["Product"]
    meter_locations = df["MeterLocation"]
    producers = df["Producer"]
    mesurement_techs = df["MeasurementTech"]
    are_opf_checks = df["IsOpfCheck"]
    dots = df["DOT"]
    order_and_types = df["OrderAndType"]
    delivery_points = df["DeliveryPoint"]
    equipment_ids = df["EquipmentID"]
    meter_names = df["MeterName"]
    routes = df["Route"]
    areas = df["Area"]

    for idx, plc_tag in enumerate(plc_tags):
        tag_type = plc_tag_types.iloc[idx]
        udt_def = udt_defs.iloc[idx]
        ign_atom = ign_atoms.iloc[idx]
        plc = plcs.iloc[idx]
        base_path = base_paths.iloc[idx]
        description = descriptions.iloc[idx]
        facility_id = facility_ids.iloc[idx]
        ign_tag_name = ign_tag_names.iloc[idx]
        ### Metadata Additions
        equip_type = (
            equip_types.iloc[idx]
            if not isinstance(equip_types.iloc[idx], float)
            else ""
        )
        facility = (
            facilities.iloc[idx] if not isinstance(facilities.iloc[idx], float) else ""
        )
        is_active = (
            are_active.iloc[idx] if not isinstance(are_active.iloc[idx], float) else ""
        )
        service = (
            services.iloc[idx] if not isinstance(services.iloc[idx], float) else ""
        )
        site = sites.iloc[idx] if not isinstance(sites.iloc[idx], float) else ""
        disposition = (
            dispositions.iloc[idx]
            if not isinstance(dispositions.iloc[idx], float)
            else ""
        )
        product = (
            products.iloc[idx] if not isinstance(products.iloc[idx], float) else ""
        )
        meter_location = (
            meter_locations.iloc[idx]
            if not isinstance(meter_locations.iloc[idx], float)
            else ""
        )
        producer = (
            producers.iloc[idx] if not isinstance(producers.iloc[idx], float) else ""
        )
        mesurement_tech = (
            mesurement_techs.iloc[idx]
            if not isinstance(mesurement_techs.iloc[idx], float)
            else ""
        )
        is_opf_check = (
            are_opf_checks.iloc[idx]
            if not isinstance(are_opf_checks.iloc[idx], float)
            else ""
        )
        dot = dots.iloc[idx] if not isinstance(dots.iloc[idx], float) else ""
        order_and_type = (
            order_and_types.iloc[idx]
            if not isinstance(order_and_types.iloc[idx], float)
            else ""
        )
        delivery_point = (
            delivery_points.iloc[idx]
            if not isinstance(delivery_points.iloc[idx], float)
            else ""
        )
        equipment_id = (
            equipment_ids.iloc[idx]
            if not isinstance(equipment_ids.iloc[idx], float)
            else ""
        )
        meter_name = (
            meter_names.iloc[idx]
            if not isinstance(meter_names.iloc[idx], float)
            else ""
        )
        route = routes.iloc[idx] if not isinstance(routes.iloc[idx], float) else ""
        area = areas.iloc[idx] if not isinstance(areas.iloc[idx], float) else ""

        if not isinstance(tag_type, str):
            continue
        if not isinstance(udt_def, str):
            continue
        if not isinstance(ign_atom, str):
            continue
        if not isinstance(plc_tag, str):
            continue
        if not isinstance(base_path, str):
            continue
        if not isinstance(ign_tag_name, str) and not isinstance(ign_tag_name, int):
            continue
        ign_tag_name = str(ign_tag_name)
        plc_tag = plc_tag.replace("{topic}.", "")
        tag_type = "Float" if tag_type == "Double" else tag_type
        tagname_idx = idx
        ign_atom = ign_atom.replace("\\", "/").replace("\n", "").strip()
        if "/" in ign_atom:
            logger.debug(
                f"{context}: Line Skipped: {tag_type=} | {udt_def=} | {ign_atom=} | {plc_tag=} |> No tags permitted on properties"
            )
            continue
        try:
            template_strategy: TemplateStrategy = udt_atomic_map[(udt_def, ign_atom)]
        except KeyError:
            logger.debug(
                f"{context}: Line Skipped: {tag_type=} | {udt_def=} | {ign_atom=} | {plc_tag=} |> could not find atomic match from strategy template"
            )
            continue

        if template_strategy.strat_class == "waste":
            logger.debug(
                f"{context}: Line Skipped: {tag_type=} | {udt_def=} | {ign_atom=} | {plc_tag=} |> strategy class sheet defines this combination as 'waste'"
            )
        Strat = template_strategy.template_strat
        params = template_strategy.parameters.values()
        if udt_def == "INTERLOCK":
            ign_tag_name += "_ITLK"
        instance_key = InstanceKey(ign_tag_name, udt_def, base_path, plc)

        try:
            strat: AtomicStrategy = Strat(  # type: ignore
                AtomicData(
                    plc_tag,
                    ign_tag_name,  # type: ignore
                    ign_atom,
                ),
                False,
                *params,
            )
        except IndexError:
            if not template_strategy.overwrite:
                raise ValueError(
                    f"{instance_key=} unauthorized break on value {plc_tag=}"
                )
            if instance_key not in instance_dict.keys():
                instance_dict[instance_key] = {  # type: ignore
                    f"{{_t_{ign_tag_name}}}": plc_tag
                }
                continue
            else:
                instance_dict[instance_key][f"{{_t_{ign_tag_name}}}"] = plc_tag
                continue
        if strat.break_strat and not template_strategy.overwrite:
            raise ValueError(f"{instance_key=} unauthorized break on value {plc_tag=}")
        if strat.break_strat and instance_key not in instance_dict.keys():
            instance_dict[instance_key] = {f"{{_t_{ign_tag_name}}}": plc_tag}
            continue
        elif strat.break_strat:
            instance_dict[instance_key][f"{{_t_{ign_tag_name}}}"] = plc_tag
            continue

        param_dict = dict(
            zip(
                strat.template_parameters,
                strat.instance_parameters,
            )
        )

        metadata_dict = {
            "Description": description,
            "FacilityID": facility_id,
            "EquipType": equip_type,
            "Facility": facility,
            "IsActive": is_active,
            "Service": service,
            "Site": site,
            "Disposition": disposition,
            "Product": product,
            "MeterLocation": meter_location,
            "Producer": producer,
            "MeasurementTech": mesurement_tech,
            "IsOpfCheck": is_opf_check,
            "DOT": dot,
            "OrderAndType": order_and_type,
            "DeliveryPoint": delivery_point,
            "EquipmentID": equipment_id,
            "MeterName": meter_name,
            "Route": route,
            "Area": area,
        }

        if instance_key not in instance_dict.keys():
            instance_dict[instance_key] = param_dict | metadata_dict
        else:
            instance_dict[instance_key] = instance_dict[instance_key] | param_dict

    return instance_dict


def create_udt_config_json(
    udt_data: UDTData,
    root_path: str,
    extended_udt_map: ExtendedUDTDict,
) -> dict:
    tag_list = []
    udt_path_dict = {}
    for udt, data in udt_data.items():
        base_path = root_path + udt.split("/")[0] + "/"
        udt_dict = {
            "name": udt.split("/")[1],
            "typeID": extended_udt_map[udt],
            "tags": [],
            "parameters": {para: {"dataType": "String"} for para in data.parameters},
            "tagType": "UdtType",
        }
        for atomic in data.atomics:
            if isinstance(atomic.opc_path, str):
                atomic.opc_path = "[{Topic}]" + atomic.opc_path
            else:
                atomic.opc_path["binding"] = "[{Topic}]" + atomic.opc_path["binding"]
            atomic_dict = {
                "name": atomic.ign_name,
                "valueSource": "opc",
                "LongID": "",
                "Tag_Description": "",
                "Safety": "",
                "PI_Tag": False,
                "DOT": False,
                "Units": "",
                "opcItemPath": atomic.opc_path,
                "historyProvider": "HistDB_Longmont",
            }
            if f"_t_{atomic.ign_name}" in udt_dict["parameters"].keys():
                udt_dict["parameters"][f"_t_{atomic.ign_name}"]["value"] = (
                    atomic.opc_path
                )
                atomic_dict["opcItemPath"] = binding(
                    f"[{{Topic}}]{{_t_{atomic.ign_name}}}"
                )
            udt_dict["tags"].append(atomic_dict)
        if base_path not in udt_path_dict.keys():
            udt_path_dict[base_path] = [udt_dict]
        else:
            udt_path_dict[base_path].append(udt_dict)
        tag_list.append(udt_dict)
    return udt_path_dict


def create_instance_config_json(
    inst_data: InstData,
    root_path: str,
    udt_root_path: str,
) -> dict:
    inst_dict = {}
    for keys, parameters in inst_data.items():
        tag_inst = keys.ign_inst_name
        udt = keys.udt_type
        folder_extension = keys.base_path
        device = keys.device
        base_udt_list = [name.replace("ALLEN_BRADLEY/", "") for name in BASE_UDT_NAME]
        udt_type = udt if udt not in base_udt_list else udt + "_STD"
        full_path = root_path + "/" + folder_extension
        if full_path not in inst_dict.keys():
            inst_dict[full_path] = {}
        if tag_inst not in inst_dict[full_path].keys():
            inst_dict[full_path][tag_inst] = {
                "name": tag_inst,
                "typeId": udt_root_path + udt_type,
                "tagType": "UdtInstance",
                "parameters": parameters | {"Topic": device},
            }
        else:
            inst_dict[full_path][tag_inst]["parameters"] = (
                inst_dict[full_path][tag_inst]["parameters"] | parameters
            )
    for key, tags in inst_dict.items():
        inst_dict[key] = [tag for tag in tags.values()]
    return inst_dict


def strategy_merge(
    old_sheet_path: str,
    old_tab_name: str,
    new_sheet_path: str,
    new_tab_name: str,
    new_output_excel: str,
    max_parameters: int,
) -> None:
    df = pd.read_excel(old_sheet_path, old_tab_name)
    old_ext_udt = df["Extended UDT"]
    old_ign_names = df["Ignition Name"]
    old_strats = df["Template Strategy"]
    old_overwrites = df["Overwrite Option"]
    para_dict = {i: df[f"Parameter {i}"] for i in range(1, max_parameters + 1)}
    old_data_dict: dict[tuple[str, str], dict[str, str]] = {}
    for idx, old_udt in enumerate(old_ext_udt):
        old_ign_name = old_ign_names.iloc[idx]
        old_strat = old_strats.iloc[idx]
        old_ow = old_overwrites.iloc[idx]
        old_para_dict = {
            "Parameter " + str(key): val[idx] for key, val in para_dict.items()
        }
        old_data_dict[(old_udt, old_ign_name)] = old_para_dict | {
            "Template Strategy": old_strat,
            "Overwrite Option": old_ow,
        }

    data = {
        "Extended UDT": [],
        "Ignition Name": [],
        "Data Type": [],
        "Atomic Type": [],
        "Sample UDT Instance Name": [],
        "Sample PLC Tag": [],
        "Template Strategy": [],
    }
    for i in range(1, max_parameters + 1):
        data[f"Parameter {i}"] = []
    data["Overwrite Option"] = []
    new_df = pd.read_excel(new_sheet_path, new_tab_name)
    new_ext_udt = new_df["Extended UDT"]
    new_ign_names = new_df["Ignition Name"]

    for idx, new_udt in enumerate(new_ext_udt):
        new_ign_name: str = new_ign_names.iloc[idx]

        data["Extended UDT"].append(new_udt)
        data["Ignition Name"].append(new_df["Ignition Name"][idx])
        data["Data Type"].append(new_df["Data Type"][idx])
        data["Atomic Type"].append(new_df["Atomic Type"][idx])
        data["Sample UDT Instance Name"].append(new_df["Sample UDT Instance Name"][idx])
        data["Sample PLC Tag"].append(new_df["Sample PLC Tag"][idx])

        if (new_udt, new_ign_names[idx]) in old_data_dict.keys():
            data["Overwrite Option"].append(
                old_data_dict[(new_udt, new_ign_name)]["Overwrite Option"]
            )
            data["Template Strategy"].append(
                old_data_dict[(new_udt, new_ign_name)]["Template Strategy"]
            )
            for i in range(1, max_parameters + 1):
                data[f"Parameter {i}"].append(
                    old_data_dict[(new_udt, new_ign_name)][f"Parameter {i}"]
                )
        else:
            data["Overwrite Option"].append("")
            data["Template Strategy"].append("")
            for i in range(1, max_parameters + 1):
                data[f"Parameter {i}"].append("")

    df = pd.DataFrame(data)
    df.to_excel(new_output_excel, index=False, engine="openpyxl")


if __name__ == "__main__":
    from pprint import pprint

    ### RESOURCE IMPORT ###
    extended_udt_map = build_extended_udt_map(xl_file_path, instance_sheet_name)
    # print('extended udt map built')
    base_tag_dict = base_tag_dict_import()
    # print('generated base tag dict')
    ### SAMPLE BUILD ###
    # sample_dict = generate_udt_atomic_samples(
    #     extended_udt_map,
    #     base_tag_dict,
    #     xl_file_path,
    #     instance_sheet_name,
    # )
    # make_xl = create_xl_strategy_book(strategy_book_file_output, sample_dict, 4)
    ### STRATEGY IMPORT ###
    strategy_sheet = StrategySheet(strategy_book_file_input, 4)
    ### UDT GENERATION ###
    udt_data: UDTData = create_udt_data(strategy_sheet, extended_udt_map)
    ### ATOMIC DATA COLLECTION ###
    atomic_check_dict = atomic_check_dict_build(udt_data, strategy_sheet)
    # ### CONFORMANCE CHECK ###
    # conformance = udt_atomic_instance_conformance_check(
    #     atomic_check_dict,
    #     xl_file_path,
    #     instance_sheet_name,
    #     5
    # )
    # pprint(conformance)
    # create_conformance_book(
    #     'conformance_book_6_14.xlsx',
    #     conformance,
    #     5,
    # )
    ### INSTANCE BUILD ###
    instance_dict: InstData = tag_instance_build(
        atomic_check_dict,
        xl_file_path,
    )
    # pprint(
    #     instance_dict
    # )
    print(len(instance_dict))
    ### GENERATE UDT JSON ###
    udt_config = create_udt_config_json(
        udt_data,
        "[Longmont]_types_/",
        extended_udt_map,
    )
    with open(udt_file_output, "w") as file:
        json.dump(udt_config, file, indent=4)
    ### GENERATE INST JSON ###
    inst_config = create_instance_config_json(
        instance_dict,
        "[Longmont]",
        "",
    )
    with open(instance_file_output, "w") as file:
        json.dump(inst_config, file, indent=4)
    # pprint(inst_config)
    ### UTILITY::: MERGE STRATEGIES TO NEW TEMPLATE
    # strategy_merge(
    #     'Longmont Template Strategy Merge 6_23.xlsx',
    #     'Sheet1',
    #     'Longmont Template Strategy out 6_23_B.xlsx',
    #     'Sheet1',
    #     'Longmont Template Strategy Merge 6_23_B.xlsx',
    #     4,
    # )
