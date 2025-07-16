from copy import deepcopy
from dataclasses import dataclass
from typing import Iterator, Literal, Protocol
import re

TYPE_OPTIONS_LIST = ['Float', 'Double', 'Integer', 'Boolean', 'String']
type TypeOptions = Literal['Float', 'Double', 'Integer', 'Boolean', 'String']

type Binding = dict[str, str]

def binding(value: str) -> Binding:
    return {
        "bindType": "parameter",
        "binding": value
    }

### DATACLASS DEFINITIONS ###
@dataclass(frozen=True)
class AtomicTag:
    name: str
    datatype: TypeOptions

@dataclass(frozen=True)
class TagSample:
    ign_name: str
    datatype: str
    atomic_type: Literal['opc', 'extension', 'waste']

@dataclass(frozen=True)
class AtomicData:
    plc_tag_name: str
    ign_inst_name: str
    ign_atomic_name: str

### PROTOCOL DEFINITIONS ###
class AtomicStrategy(Protocol):
    atomic_data: AtomicData
    overwrite: bool
    break_strat: bool = False

    def __init__(self, atomic_data: AtomicData, overwrite: bool = False):
        self.atomic_data = atomic_data
        self.overwrite = overwrite

    @property
    def template(self) -> str | Binding:
        ...

    @property
    def template_parameters(self) -> Iterator[str]:
        ...

    @property
    def instance_parameters(self) -> Iterator[str]:
        ...


### STRATEGIES ###
class Static(AtomicStrategy):
    @property
    def template(self) -> str:
        return self.atomic_data.plc_tag_name
    
    @property
    def template_parameters(self) -> Iterator[str]:
        yield ''

    @property
    def instance_parameters(self) -> Iterator[str]:
        yield ''

class ParentName(AtomicStrategy):
    @property
    def template(self) -> Binding:
        return binding(self.atomic_data.plc_tag_name.replace(
            self.atomic_data.ign_inst_name, 
            '{ParentInstanceName}',
            )
        )
    
    @property
    def template_parameters(self) -> Iterator[str]:
        yield ''

    @property
    def instance_parameters(self) -> Iterator[str]:
        yield ''

class StaticOverwrite(AtomicStrategy):
    overwrite = True
    @property
    def template(self) -> Binding:
        return binding("")

    @property
    def template_parameters(self) -> Iterator[str]:
        yield f"_t_{self.atomic_data.ign_atomic_name}"

    @property
    def instance_parameters(self) -> Iterator[str]:
        yield self.atomic_data.plc_tag_name

class UnderscoreSplitIndex(AtomicStrategy):
    def __init__(
            self,
            atomic_data,
            overwrite,
            type_label,
            number_label,
            type_index,
            number_index,
        ) -> None:
        super().__init__(atomic_data, overwrite)
        self.type_label = f'_p_{type_label}'
        self.number_label = f'_p_{number_label}'
        self.type_index = type_index
        self.number_index = number_index
        template: list[str] = []
        try:
            template = self.atomic_data.plc_tag_name.split('_')
        except:
            self.break_strat = True
        self.instance_type = deepcopy(template[self.type_index])
        self.instance_number = deepcopy(template[self.number_index])
        if not self.break_strat:
            template[self.type_index] = f'{{{self.type_label}}}'
            template[self.number_index] = f'{{{self.number_label}}}'
        self.template_val = '_'.join(template)

    @property
    def template(self) -> Binding:
        return binding(self.template_val)

    @property
    def template_parameters(self) -> Iterator[str]:
        yield self.type_label
        yield self.number_label
        if self.overwrite:
            yield f'_t_{self.atomic_data.ign_atomic_name}'

    @property
    def instance_parameters(self) -> Iterator[str]:
        yield self.instance_type
        yield self.instance_number


class UnderscoreDotSplit(AtomicStrategy):
    def __init__(
            self,
            atomic_data,
            overwrite,
            type_label,
            number_label,
        ) -> None:
        super().__init__(atomic_data, overwrite)
        self.type_label = f'_p_{type_label}'
        self.number_label = f'_p_{number_label}'
        u_split: list[str] = []
        try:
            u_split = self.atomic_data.plc_tag_name.split('_')
        except:
            self.break_strat = True
        self.instance_type = deepcopy(u_split[0])
        dot_split: list[str] = []
        try: 
            dot_split = u_split[1].split('.')
        except:
            self.break_strat = True
        self.instance_number = deepcopy(dot_split[0])
        if not self.break_strat:
            try:
                self.template_val = f'{{{self.type_label}}}_{{{self.number_label}}}.{dot_split[1]}'
            except IndexError:
                self.template_val = f'{{{self.type_label}}}_{{{self.number_label}}}'
        else:
            self.template_val = ''

    @property
    def template(self) -> Binding:
        return binding(self.template_val)

    @property
    def template_parameters(self) -> Iterator[str]:
        yield self.type_label
        yield self.number_label
        if self.overwrite:
            yield f'_t_{self.atomic_data.ign_atomic_name}'

    @property
    def instance_parameters(self) -> Iterator[str]:
        yield self.instance_type
        yield self.instance_number

class NameDotSplit(AtomicStrategy):
    def __init__(
        self,
        atomic_data,
        overwrite,
        name_label,
        ) -> None:
        super().__init__(atomic_data, overwrite)
        self.name_label = f'_p_{name_label}'
        dot_split: list[str] = []
        try:
            dot_split = self.atomic_data.plc_tag_name.split('.')
        except:
            self.break_strat = True
        self.instance_name = deepcopy(dot_split[0])
        if not self.break_strat:
            dot_split[0] = f'{{{self.name_label}}}'
        self.template_val = '.'.join(dot_split)
    
    @property
    def template(self) -> Binding:
        return binding(self.template_val)

    @property
    def template_parameters(self) -> Iterator[str]:
        yield self.name_label
        if self.overwrite:
            yield f'_t_{self.atomic_data.ign_atomic_name}'

    @property
    def instance_parameters(self) -> Iterator[str]:
        yield self.instance_name

class NameLastUnderscoreSplit(AtomicStrategy):
    def __init__(
        self,
        atomic_data,
        overwrite,
        name_label,
        ) -> None:
        super().__init__(atomic_data, overwrite)
        self.name_label = f'_p_{name_label}'
        under_split: list[str] = []
        try:
            under_split = self.atomic_data.plc_tag_name.split('_')
        except:
            self.break_strat = True
        self.instance_name = deepcopy('_'.join(under_split[:-1]))
        self.template_val = f'{{{self.name_label}}}' + under_split[-1]

    @property
    def template(self) -> Binding:
        return binding(self.template_val)

    @property
    def template_parameters(self) -> Iterator[str]:
        yield self.name_label
        if self.overwrite:
            yield f'_t_{self.atomic_data.ign_atomic_name}'

    @property
    def instance_parameters(self) -> Iterator[str]:
        yield self.instance_name

class LubeInstrumentSplit(AtomicStrategy):
    def __init__(
        self,
        atomic_data,
        overwrite,
        ) -> None:
        super().__init__(atomic_data, overwrite)
        self.analog_type = '{_p_Analog Type}'
        self.analog_number = '{_p_Analog Number}'

        u_split: list[str] = []
        try:
            u_split = self.atomic_data.plc_tag_name.split('_')
        except:
            self.break_strat = True
        self.instance_val = deepcopy(u_split[-2])
        self.instance_type = self.instance_val[0:3]
        self.instance_num = self.instance_val[2:]
        if not self.break_strat:
            u_split[-2] = f'{{{self.analog_type}}}{{{self.analog_number}}}'
        self.template_val = '_'.join(u_split)
    
    @property
    def template(self) -> Binding:
        return binding(self.template_val)

    @property
    def template_parameters(self) -> Iterator[str]:
        yield self.analog_type
        yield self.analog_number
        if self.overwrite:
            yield f'_t_{self.atomic_data.ign_atomic_name}'

    @property
    def instance_parameters(self) -> Iterator[str]:
        yield self.instance_type
        yield self.instance_num

class LubeDotSplit(AtomicStrategy):
    def __init__(
        self,
        atomic_data,
        overwrite,
        ) -> None:
        super().__init__(atomic_data, overwrite)
        self.process_type = '_p_Process Type'
        u_split: list[str] = []
        dot_split: list[str] = []
        try:
            u_split = self.atomic_data.plc_tag_name.split('_')
            dot_split = u_split[-1].split('.')
        except:
            self.break_strat = True
        self.instance_val = deepcopy(dot_split[0])
        if not self.break_strat:
            dot_split[0] = f'{{{self.process_type}}}'
            dot_join = '.'.join(dot_split)
            u_split[-1] = dot_join
        u_join = '_'.join(u_split)
        self.template_val = u_join
    
    @property
    def template(self) -> Binding:
        return binding(self.template_val)

    @property
    def template_parameters(self) -> Iterator[str]:
        yield self.process_type
        if self.overwrite:
            yield f'_t_{self.atomic_data.ign_atomic_name}'

    @property
    def instance_parameters(self) -> Iterator[str]:
        yield self.instance_val

class LubeUnderSplit(AtomicStrategy):
    def __init__(
        self,
        atomic_data,
        overwrite,
        ) -> None:
        super().__init__(atomic_data, overwrite)
        self.process_type = '_p_Process Type'
        u_split: list[str] = []
        try:
            u_split = self.atomic_data.plc_tag_name.split('_')
        except:
            self.break_strat = True
        self.instance_val = deepcopy(u_split[-2]) if not self.break_strat else ''
        if not self.break_strat:
            u_split[-2] = f'{{{self.process_type}}}'
        u_join = '_'.join(u_split)
        self.template_val = u_join
    
    @property
    def template(self) -> Binding:
        return binding(self.template_val)

    @property
    def template_parameters(self) -> Iterator[str]:
        yield self.process_type
        if self.overwrite:
            yield f'_t_{self.atomic_data.ign_atomic_name}'

    @property
    def instance_parameters(self) -> Iterator[str]:
        yield self.instance_val

# PT2425A.OOR_ALM | Label Name | Label Number | Assumes Label Name is Left of Split
class NumberDotSplit(AtomicStrategy):
    def __init__(
        self,
        atomic_data,
        overwrite,
        name_label,
        number_label,
        ):
        super().__init__(atomic_data, overwrite)
        self.name_label = f'_p_{name_label}'
        self.number_label = f'_p_{number_label}'
        numbers = re.findall(r'\d+', self.atomic_data.plc_tag_name)
        if not numbers:
            self.break_strat = True
        number_split = self.atomic_data.plc_tag_name.split(numbers[0])
        dot_split = number_split[1].split('.')
        self.instance_name = deepcopy(number_split[0])
        # Check '.' exists
        if len(dot_split) > 1:
            self.instance_number = deepcopy(numbers[0]) + deepcopy(dot_split[0])
            self.template_val = f'{{{self.name_label}}}{{{self.number_label}}}.{dot_split[1]}'
        else:
            self.instance_number = deepcopy(numbers[0])
            self.template_val = f'{{{self.name_label}}}{{{self.number_label}}}{number_split[1]}'

    @property
    def template(self) -> Binding:
        return binding(self.template_val)

    @property
    def template_parameters(self) -> Iterator[str]:
        yield self.name_label
        yield self.number_label
        if self.overwrite:
            yield f'_t_{self.atomic_data.ign_atomic_name}'

    @property
    def instance_parameters(self) -> Iterator[str]:
        yield self.instance_number
        yield self.instance_name

# 'NumberUnderscoreSplit': (NumberUnderscoreSplit, 2), # PIC3010A_Selected
class NumberUnderscoreSplit(AtomicStrategy):
    def __init__(
        self,
        atomic_data,
        overwrite,
        name_label,
        number_label,
        ):
        super().__init__(atomic_data, overwrite)
        self.name_label = f'_p_{name_label}'
        self.number_label = f'_p_{number_label}'
        numbers = re.findall(r'\d+', self.atomic_data.plc_tag_name)
        if not numbers:
            self.break_strat = True
        number_split = self.atomic_data.plc_tag_name.split(numbers[0])
        dot_split = number_split[1].split('_')
        self.instance_name = deepcopy(number_split[0])
        # Check '.' exists
        if len(dot_split) > 1:
            self.instance_number = deepcopy(numbers[0]) + deepcopy(dot_split[0])
            self.template_val = f'{{{self.name_label}}}{{{self.number_label}}}_{dot_split[1]}'
        else:
            self.instance_number = deepcopy(numbers[0])
            self.template_val = f'{{{self.name_label}}}{{{self.number_label}}}{number_split[1]}'

    @property
    def template(self) -> Binding:
        return binding(self.template_val)

    @property
    def template_parameters(self) -> Iterator[str]:
        yield self.name_label
        yield self.number_label
        if self.overwrite:
            yield f'_t_{self.atomic_data.ign_atomic_name}'

    @property
    def instance_parameters(self) -> Iterator[str]:
        yield self.instance_number
        yield self.instance_name

# HS_71209A | Label Number
class NumberOnlySplit(AtomicStrategy):
    def __init__(
        self,
        atomic_data,
        overwrite,
        name_label,
        number_label,
        ):
        super().__init__(atomic_data, overwrite)
        self.name_label = f'_p_{name_label}'
        self.number_label = f'_p_{number_label}'
        numbers = re.findall(r'\d+', self.atomic_data.plc_tag_name)
        if not numbers:
            self.break_strat = True
        number_split = self.atomic_data.plc_tag_name.split(numbers[0])
        self.template_val = f'{number_split[0]}{{{self.number_label}}}'
        self.instance_number = str(numbers[0]) if len(number_split) == 1 else str(numbers[0]) + number_split[1]

    @property
    def template(self) -> Binding:
        return binding(self.template_val)

    @property
    def template_parameters(self) -> Iterator[str]:
        yield self.number_label
        if self.overwrite:
            yield f'_t_{self.atomic_data.ign_atomic_name}'

    @property
    def instance_parameters(self) -> Iterator[str]:
        yield self.instance_number


class NameUnderscoreSplitIndexJoin(AtomicStrategy):
    def __init__(
        self,
        atomic_data,
        overwrite,
        name_label,
        split_index,
        ) -> None:
        super().__init__(atomic_data, overwrite)
        self.name_label = f'_p_{name_label}'
        under_split: list[str] = []
        try:
            under_split = self.atomic_data.plc_tag_name.split('_')
        except:
            self.break_strat = True
        self.instance_name = deepcopy('_'.join(under_split[:split_index + 1]))
        self.template_val = f'{{{self.name_label}}}' + '_'.join(under_split[split_index + 1:])

    @property
    def template(self) -> Binding:
        return binding(self.template_val)

    @property
    def template_parameters(self) -> Iterator[str]:
        yield self.name_label
        if self.overwrite:
            yield f'_t_{self.atomic_data.ign_atomic_name}'

    @property
    def instance_parameters(self) -> Iterator[str]:
        yield self.instance_name

class Rejected(AtomicStrategy):
    @property
    def template(self) -> Binding:
        return binding('')
    @property
    def template_parameters(self) -> Iterator[str]:
        yield ''

    @property
    def instance_parameters(self) -> Iterator[str]:
        yield ''

type AtomicStrategyString = Literal[
    'Static',
    'ParentName',
    'StaticOverwrite',
    'UnderscoreSplitIndex',
    'UnderscoreDotSplit',
]
type Parameters = int
type AtomicStratDict = dict[AtomicStrategyString, tuple[AtomicStrategy,Parameters]]


ATOMIC_STRAT_DICT: AtomicStratDict = {
    'Static': (Static, 0),
    'ParentName': (ParentName, 0),
    'StaticOverwrite': (StaticOverwrite, 0),
    'UnderscoreSplitIndex': (UnderscoreSplitIndex, 4),
    'UnderscoreDotSplit': (UnderscoreDotSplit, 2),
    'NameDotSplit': (NameDotSplit, 1),
    'NameLastUnderscoreSplit': (NameLastUnderscoreSplit, 1),
    'LubeInstrumentSplit': (LubeInstrumentSplit, 0),
    'LubeDotSplit': (LubeDotSplit, 0),
    'LubeUnderSplit': (LubeUnderSplit, 0),
    # New Adds
    'NumberDotSplit': (NumberDotSplit, 2), # PT2425A.OOR_ALM
    'NumberOnlySplit': (NumberOnlySplit, 2), # HS_71209A
    'NumberUnderscoreSplit': (NumberUnderscoreSplit, 2), # PIC3010A_Selected
    'NameUnderscoreSplitIndexJoin': (NameUnderscoreSplitIndexJoin, 2), # LOUVER_PID_ProgManReq
    'Rejected': (Rejected, 0), # ---
    'Reject': (Rejected, 0), # ---
}
