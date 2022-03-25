# Copyright 2019 The Meson development team

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This class contains the basic functionality needed to run any interpreter
# or an interpreter-based tool.

from .common import CMakeException
from .generator import parse_generator_expressions
from .. import mlog
from ..mesonlib import version_compare

import typing as T
from pathlib import Path
from functools import lru_cache
import re
import json
import textwrap

if T.TYPE_CHECKING:
    from ..environment import Environment

class CMakeTraceLine:
    def __init__(self, file_str: str, line: int, func: str, args: T.List[str]) -> None:
        self.file = CMakeTraceLine._to_path(file_str)
        self.line = line
        self.func = func.lower()
        self.args = args

    @staticmethod
    @lru_cache(maxsize=None)
    def _to_path(file_str: str) -> Path:
        return Path(file_str)

    def __repr__(self) -> str:
        s = 'CMake TRACE: {0}:{1} {2}({3})'
        return s.format(self.file, self.line, self.func, self.args)

class CMakeCacheEntry(T.NamedTuple):
    value: T.List[str]
    type: str

class CMakeTarget:
    def __init__(
                self,
                name:        T.Optional[str],
                target_type: str,
                properties:  T.Optional[T.Dict[str, T.List[str]]] = None,
                imported:    bool                                 = False,
                tline:       T.Optional[CMakeTraceLine]           = None
            ):
        if properties is None:
            properties = {}
        self.name            = name
        self.type            = target_type
        self.properties      = properties
        self.imported        = imported
        self.tline           = tline
        self.depends         = []      # type: T.List[str]
        self.current_bin_dir = None    # type: T.Optional[Path]
        self.current_src_dir = None    # type: T.Optional[Path]

    def __repr__(self) -> str:
        s = 'CMake TARGET:\n  -- name:      {}\n  -- type:      {}\n  -- imported:  {}\n  -- properties: {{\n{}     }}\n  -- tline: {}'
        propSTR = ''
        for i in self.properties:
            propSTR += "      '{}': {}\n".format(i, self.properties[i])
        return s.format(self.name, self.type, self.imported, propSTR, self.tline)

    def strip_properties(self) -> None:
        # Strip the strings in the properties
        if not self.properties:
            return
        for key, val in self.properties.items():
            self.properties[key] = [x.strip() for x in val]
            assert all(';' not in x for x in self.properties[key])

class CMakeGeneratorTarget(CMakeTarget):
    def __init__(self, name: T.Optional[str]) -> None:
        super().__init__(name, 'CUSTOM', {})
        self.outputs = []        # type: T.List[Path]
        self.command = []        # type: T.List[T.List[str]]
        self.working_dir = None  # type: T.Optional[Path]

class CMakeTraceParser:
    def __init__(self, cmake_version: str, build_dir: Path, env: 'Environment', permissive: bool = True) -> None:
        self.vars:                      T.Dict[str, T.List[str]]     = {}
        self.vars_by_file: T.Dict[Path, T.Dict[str, T.List[str]]]    = {}
        self.targets:                   T.Dict[str, CMakeTarget]     = {}
        self.cache:                     T.Dict[str, CMakeCacheEntry] = {}

        self.explicit_headers = set()  # type: T.Set[Path]

        # T.List of targes that were added with add_custom_command to generate files
        self.custom_targets = []  # type: T.List[CMakeGeneratorTarget]

        self.env = env
        self.permissive = permissive  # type: bool
        self.cmake_version = cmake_version  # type: str
        self.trace_file = 'cmake_trace.txt'
        self.trace_file_path = build_dir / self.trace_file
        self.trace_format = 'json-v1' if version_compare(cmake_version, '>=3.17') else 'human'

        # State for delayed command execution. Delayed command execution is realised
        # with a custom CMake file that overrides some functions and adds some
        # introspection information to the trace.
        self.delayed_commands = []  # type: T.List[str]
        self.stored_commands = []   # type: T.List[CMakeTraceLine]

        # All supported functions
        self.functions: T.Dict[str, T.Callable[[CMakeTraceLine], None]] = {
            'set': self._cmake_set,
            'unset': self._cmake_unset,
            'string': self._cmake_string,
            'list': self._cmake_list,
            'add_executable': self._cmake_add_executable,
            'add_library': self._cmake_add_library,
            'add_custom_command': self._cmake_add_custom_command,
            'add_custom_target': self._cmake_add_custom_target,
            'set_property': self._cmake_set_property,
            'set_target_properties': self._cmake_set_target_properties,
            'target_compile_definitions': self._cmake_target_compile_definitions,
            'target_compile_options': self._cmake_target_compile_options,
            'target_include_directories': self._cmake_target_include_directories,
            'target_link_libraries': self._cmake_target_link_libraries,
            'target_link_options': self._cmake_target_link_options,
            'add_dependencies': self._cmake_add_dependencies,

            # Special functions defined in the preload script.
            # These functions do nothing in the CMake code, but have special
            # meaning here in the trace parser.
            'meson_ps_execute_delayed_calls': self._meson_ps_execute_delayed_calls,
            'meson_ps_reload_vars': self._meson_ps_reload_vars,
            'meson_ps_disabled_function': self._meson_ps_disabled_function,
        }

        if version_compare(self.cmake_version, '<3.17.0'):
            mlog.deprecation(textwrap.dedent(f'''\
                CMake support for versions <3.17 is deprecated since Meson 0.62.0.
                |
                |   However, Meson was only able to find CMake {self.cmake_version}.
                |
                |   Support for all CMake versions below 3.17.0 will be removed once
                |   newer CMake versions are more widely adopted. If you encounter
                |   any errors please try upgrading CMake to a newer version first.
            '''), once=True)

    def trace_args(self) -> T.List[str]:
        arg_map = {
            'human': ['--trace', '--trace-expand'],
            'json-v1': ['--trace-expand', '--trace-format=json-v1'],
        }

        base_args = ['--no-warn-unused-cli']
        if not self.requires_stderr():
            base_args += [f'--trace-redirect={self.trace_file}']

        return arg_map[self.trace_format] + base_args

    def requires_stderr(self) -> bool:
        return version_compare(self.cmake_version, '<3.16')

    def parse(self, trace: T.Optional[str] = None) -> None:
        # First load the trace (if required)
        if not self.requires_stderr():
            if not self.trace_file_path.exists and not self.trace_file_path.is_file():
                raise CMakeException(f'CMake: Trace file "{self.trace_file_path!s}" not found')
            trace = self.trace_file_path.read_text(errors='ignore', encoding='utf-8')
        if not trace:
            raise CMakeException('CMake: The CMake trace was not provided or is empty')

        # Second parse the trace
        lexer1 = None
        if self.trace_format == 'human':
            lexer1 = self._lex_trace_human(trace)
        elif self.trace_format == 'json-v1':
            lexer1 = self._lex_trace_json(trace)
        else:
            raise CMakeException(f'CMake: Internal error: Invalid trace format {self.trace_format}. Expected [human, json-v1]')

        # Primary pass -- parse everything
        for l in lexer1:
            # store the function if its execution should be delayed
            if l.func in self.delayed_commands:
                self.stored_commands += [l]
                continue

            # "Execute" the CMake function if supported
            fn = self.functions.get(l.func, None)
            if fn:
                fn(l)

        # Evaluate generator expressions
        strlist_gen:  T.Callable[[T.List[str]],  T.List[str]]  = lambda strlist: [parse_generator_expressions(x, self) for x in strlist]
        pathlist_gen: T.Callable[[T.List[Path]], T.List[Path]] = lambda plist:   [Path(parse_generator_expressions(str(x), self)) for x in plist]

        self.vars = {k: strlist_gen(v) for k, v in self.vars.items()}
        self.vars_by_file = {
            p: {k: strlist_gen(v) for k, v in d.items()}
            for p, d in self.vars_by_file.items()
        }
        self.explicit_headers = set(Path(parse_generator_expressions(str(x), self)) for x in self.explicit_headers)
        self.cache = {
            k: CMakeCacheEntry(
                strlist_gen(v.value),
                v.type
            )
            for k, v in self.cache.items()
        }

        for tgt in self.targets.values():
            tgtlist_gen: T.Callable[[T.List[str], CMakeTarget],  T.List[str]] = lambda strlist, t: [parse_generator_expressions(x, self, context_tgt=t) for x in strlist]
            if tgt.name is not None:
                tgt.name = parse_generator_expressions(tgt.name, self, context_tgt=tgt)
            tgt.type = parse_generator_expressions(tgt.type, self, context_tgt=tgt)
            tgt.properties = {
                k: tgtlist_gen(v, tgt) for k, v in tgt.properties.items()
            }
            tgt.depends = tgtlist_gen(tgt.depends, tgt)

        for ctgt in self.custom_targets:
            ctgt.outputs = pathlist_gen(ctgt.outputs)
            ctgt.command = [strlist_gen(x) for x in ctgt.command]
            ctgt.working_dir = Path(parse_generator_expressions(str(ctgt.working_dir), self)) if ctgt.working_dir is not None else None

        # Postprocess
        for tgt in self.targets.values():
            tgt.strip_properties()

    @T.overload
    def get_cmake_list_var(
        self, file: Path, identifier: str, default: T.List[str]
    ) -> T.Tuple[T.List[str], T.List[str]]: ...

    @T.overload
    def get_cmake_list_var(
        self, file: Path, identifier: str, default: T.Optional[T.List[str]] = None
    ) -> T.Tuple[T.Optional[T.List[str]], T.Optional[T.List[str]]]: ...

    @T.overload
    def get_cmake_list_var(
        self, file: None, identifier: str, default: T.List[str]
    ) -> T.Tuple[T.List[str], None]: ...

    @T.overload
    def get_cmake_list_var(
        self, file: None, identifier: str, default: T.Optional[T.List[str]] = None
    ) -> T.Tuple[T.Optional[T.List[str]], None]: ...

    def get_cmake_list_var(
        self, file: T.Optional[Path], identifier: str, default: T.Optional[T.List[str]] = None
    ) -> T.Tuple[T.Optional[T.List[str]], T.Optional[T.List[str]]]:
        value = self.vars.get(identifier, default)
        value_by_file = None if file is None else \
            self.vars_by_file.setdefault(file, {}).get(identifier, default)
        return value, value_by_file

    @T.overload
    def set_cmake_list_var(
        self, file: Path, identifier: str, value: T.Optional[T.List[str]], value_by_file: T.Optional[T.List[str]]
    ) -> None: ...

    @T.overload
    def set_cmake_list_var(
        self, file: None, identifier: str, value: T.Optional[T.List[str]], value_by_file: None
    ) -> None: ...

    def set_cmake_list_var(
        self,
        file: T.Optional[Path],
        identifier: str,
        value: T.Optional[T.List[str]],
        value_by_file: T.Optional[T.List[str]],
    ) -> None:
        vars = self.vars
        if value is None:
            if identifier in vars:
                del vars[identifier]
        else:
            vars[identifier] = value

        if file is not None:
            vars_by_file = self.vars_by_file.setdefault(file, {})
            if value_by_file is None:
                if identifier in vars_by_file:
                    del vars_by_file[identifier]
            else:
                vars_by_file[identifier] = value_by_file

    def _list_to_str(self, value: T.List[str]) -> str:
        return ';'.join(value)

    def _str_to_list(self, value: str) -> T.List[str]:
        return value.split(';')

    @T.overload
    def get_cmake_str_var(
        self, file: Path, identifier: str, default: str
    ) -> T.Tuple[str, str]: ...

    @T.overload
    def get_cmake_str_var(
        self, file: Path, identifier: str, default: T.Optional[str] = None
    ) -> T.Tuple[T.Optional[str], T.Optional[str]]: ...

    @T.overload
    def get_cmake_str_var(
        self, file: None, identifier: str, default: str
    ) -> T.Tuple[str, None]: ...

    @T.overload
    def get_cmake_str_var(
        self, file: None, identifier: str, default: T.Optional[str] = None
    ) -> T.Tuple[T.Optional[str], None]: ...

    def get_cmake_str_var(
        self, file: T.Optional[Path], identifier: str, default: T.Optional[str] = None
    ) -> T.Tuple[T.Optional[str], T.Optional[str]]:
        list_default = None if default is None else self._str_to_list(default)
        value, value_by_file = self.get_cmake_list_var(file, identifier, list_default)
        str_value = None if not value else self._list_to_str(value)
        str_value_by_file = None if not value_by_file else self._list_to_str(value_by_file)
        return str_value, str_value_by_file

    @T.overload
    def set_cmake_str_var(
        self, file: Path, identifier: str, value: T.Optional[str], value_by_file: T.Optional[str]
    ) -> None: ...

    @T.overload
    def set_cmake_str_var(
        self, file: None, identifier: str, value: T.Optional[str], value_by_file: None
    ) -> None: ...

    def set_cmake_str_var(
        self, file: T.Optional[Path], identifier: str, value: T.Optional[str], value_by_file: T.Optional[str]
    ) -> None:
        str_value = None if value is None else self._str_to_list(value)
        if file is None:
            return self.set_cmake_list_var(file, identifier, str_value, None)
        else:
            str_value_by_file = None if value_by_file is None else self._str_to_list(value_by_file)
            return self.set_cmake_list_var(file, identifier, str_value, str_value_by_file)

    def _list_to_bool(self, value: T.List[str]) -> bool:
        if len(value) < 1:
            return False
        str_value = value[0].upper()
        return str_value not in ['0', 'OFF', 'NO', 'FALSE', 'N', 'IGNORE'] and not str_value.endswith('NOTFOUND')

    def _bool_to_list(self, value: bool) -> T.List[str]:
        if not value:
            return ['0']
        return ['1']

    @T.overload
    def get_cmake_bool_var(
        self, file: Path, identifier: str, default: bool
    ) -> T.Tuple[bool, bool]: ...

    @T.overload
    def get_cmake_bool_var(
        self, file: Path, identifier: str, default: T.Optional[bool] = None
    ) -> T.Tuple[T.Optional[bool], T.Optional[bool]]: ...

    @T.overload
    def get_cmake_bool_var(
        self, file: None, identifier: str, default: bool
    ) -> T.Tuple[bool, None]: ...

    @T.overload
    def get_cmake_bool_var(
        self, file: None, identifier: str, default: T.Optional[bool] = None
    ) -> T.Tuple[T.Optional[bool], None]: ...

    def get_cmake_bool_var(
        self, file: T.Optional[Path], identifier: str, default: T.Optional[bool] = None
    ) -> T.Tuple[T.Optional[bool], T.Optional[bool]]:
        list_default = None if default is None else self._bool_to_list(default)
        value, value_by_file = self.get_cmake_list_var(file, identifier, list_default)
        bool_value = None if value is None else self._list_to_bool(value)
        bool_value_by_file = None if value_by_file is None else self._list_to_bool(value_by_file)
        return bool_value, bool_value_by_file

    @T.overload
    def set_cmake_bool_var(
        self, file: Path, identifier: str, value: T.Optional[bool], value_by_file: T.Optional[bool]
    ) -> None: ...

    @T.overload
    def set_cmake_bool_var(
        self, file: None, identifier: str, value: T.Optional[bool], value_by_file: None
    ) -> None: ...

    def set_cmake_bool_var(
        self, file: T.Optional[Path], identifier: str, value: T.Optional[bool], value_by_file: T.Optional[bool]
    ) -> None:
        list_value = None if value is None else self._bool_to_list(value)
        if file is None:
            return self.set_cmake_list_var(file, identifier, list_value, None)
        else:
            list_value_by_file = None if value_by_file is None else self._bool_to_list(value_by_file)
            return self.set_cmake_list_var(file, identifier, list_value, list_value_by_file)

    def _gen_exception(self, function: str, error: str, tline: CMakeTraceLine) -> None:
        # Generate an exception if the parser is not in permissive mode

        if self.permissive:
            mlog.debug(f'CMake trace warning: {function}() {error}\n{tline}')
            return None
        raise CMakeException(f'CMake: {function}() {error}\n{tline}')

    def _cmake_set(self, tline: CMakeTraceLine) -> None:
        """Handler for the CMake set() function in all variants.

        Comes in three variants:
        set(<var> <value> [PARENT_SCOPE])
        set(<var> <value> CACHE <type> <docstring> [FORCE])
        set(ENV{<var>} <value>)

        We don't support the ENV variant, and any uses of it will be ignored
        silently. The other two variants are supported, with some caveats:
        - We don't properly handle scoping, so calls to set() inside a
          function without PARENT_SCOPE set could incorrectly shadow the
          outer scope.
        - We don't honor the type of CACHE arguments.
        """
        # DOC: https://cmake.org/cmake/help/latest/command/set.html

        cache_idx = None
        cache_type = None
        cache_force = False
        try:
            cache_idx = tline.args.index('CACHE')
            cache_type = tline.args[cache_idx + 1]
        except (ValueError, IndexError):
            pass

        args = list(tline.args)
        if cache_type:
            if args[-1:] == ['FORCE']:
                args.pop(-1)
                cache_force = True
            del args[cache_idx:]
        else:
            if args[-1:] == ['PARENT_SCOPE']:
                args.pop(-1)

        if len(args) < 1:
            return self._gen_exception('set', 'requires at least one argument', tline)
        identifier = args.pop(0)
        if len(args) < 1:
            value = None
        else:
            value = self._list_to_str(args)
            # Move writing to cache into `set_cmake_str_var()` so this can be removed?
            value = self._str_to_list(value)

        # Write to the CMake cache instead
        if cache_type:
            # Honor how the CMake FORCE parameter works
            if identifier not in self.cache or cache_force:
                self.cache[identifier] = CMakeCacheEntry(value if value is not None else [], cache_type)

        self.set_cmake_list_var(tline.file, identifier, value, value)

    def _cmake_unset(self, tline: CMakeTraceLine) -> None:
        # DOC: https://cmake.org/cmake/help/latest/command/unset.html
        if len(tline.args) < 1:
            return self._gen_exception('unset', 'requires at least one argument', tline)

        identifier = tline.args[0]
        self.set_cmake_list_var(tline.file, identifier, None, None)

    def _cmake_string(self, tline: CMakeTraceLine) -> None:
        """Handler for the CMake string() function in all variants.

        Comes in twenty-nine variants:
        string(FIND <string> <substring> <out-var> [...])
        string(REPLACE <match-string> <replace-string> <out-var> <input>...)
        string(REGEX MATCH <match-regex> <out-var> <input>...)
        string(REGEX MATCHALL <match-regex> <out-var> <input>...)
        string(REGEX REPLACE <match-regex> <replace-expr> <out-var> <input>...)
        string(APPEND <string-var> [<input>...])
        string(PREPEND <string-var> [<input>...])
        string(CONCAT <out-var> [<input>...])
        string(JOIN <glue> <out-var> [<input>...])
        string(TOLOWER <string> <out-var>)
        string(TOUPPER <string> <out-var>)
        string(LENGTH <string> <out-var>)
        string(SUBSTRING <string> <begin> <length> <out-var>)
        string(STRIP <string> <out-var>)
        string(GENEX_STRIP <string> <out-var>)
        string(REPEAT <string> <count> <out-var>)
        string(COMPARE <op> <string1> <string2> <out-var>)
        string(<HASH> <out-var> <input>)
        string(ASCII <number>... <out-var>)
        string(HEX <string> <out-var>)
        string(CONFIGURE <string> <out-var> [...])
        string(MAKE_C_IDENTIFIER <string> <out-var>)
        string(RANDOM [<option>...] <out-var>)
        string(TIMESTAMP <out-var> [<format string>] [UTC])
        string(UUID <out-var> ...)
        string(JSON <out-var> [ERROR_VARIABLE <error-var>]
               {GET | TYPE | LENGTH | REMOVE}
               <json-string> <member|index> [<member|index> ...])
        string(JSON <out-var> [ERROR_VARIABLE <error-var>]
               MEMBER <json-string>
               [<member|index> ...] <index>)
        string(JSON <out-var> [ERROR_VARIABLE <error-var>]
               SET <json-string>
               <member|index> [<member|index> ...] <value>)
        string(JSON <out-var> [ERROR_VARIABLE <error-var>]
               EQUAL <json-string1> <json-string2>)
        string(FIND <string> <substring> <out-var> [...])

        We only support the following variants:
        - FIND
        - REPLACE
        - APPEND
        - PREPEND
        - TOLOWER
        - TOUPPER
        - SUBSTRING
        - STRIP
        - COMPARE LESS
        - COMPARE GREATER
        - COMPARE EQUAL
        - COMPARE NOTEQUAL
        - COMPARE LESS_EQUAL
        - COMPARE GREATER_EQUAL
        Any uses of other variants will be ignored silently. The supported
        variants have some caveats:
        - We don't properly handle scoping, so calls to string() inside a
          function could incorrectly shadow the outer scope.
        """
        # DOC: https://cmake.org/cmake/help/latest/command/string.html

        args = list(tline.args)

        if len(args) < 1:
            return self._gen_exception('string', 'requires at least one argument', tline)

        variant = args.pop(0)
        if variant == 'FIND':
            value = args.pop(0).encode()
            sub_value = args.pop(0).encode()
            output_identifier = args.pop(0)
            reverse = False
            if args[-1:] == ['REVERSE']:
                reverse = True
                args.pop(-1)

            output_value = value.find(sub_value) if not reverse else value.rfind(sub_value)
            output_value = str(output_value)
            self.set_cmake_str_var(tline.file, output_identifier, output_value, output_value)
        elif variant == 'REPLACE':
            match_value = args.pop(0)
            replace_value = args.pop(0)
            output_identifier = args.pop(0)
            value = ''.join(args)

            output_value = value.replace(match_value, replace_value)
            self.set_cmake_str_var(tline.file, output_identifier, output_value, output_value)
        elif variant == 'APPEND':
            identifier = args.pop(0)
            if len(args) > 0:
                input_value = ''.join(args)

                value, value_by_file = self.get_cmake_str_var(tline.file, identifier, '')
                value += input_value
                value_by_file += input_value
                self.set_cmake_str_var(tline.file, identifier, value, value_by_file)
        elif variant == 'PREPEND':
            identifier = args.pop(0)
            if len(args) > 0:
                input_value = ''.join(args)

                value, value_by_file = self.get_cmake_str_var(tline.file, identifier, '')
                value = input_value + value
                value_by_file = input_value + value_by_file
                self.set_cmake_str_var(tline.file, identifier, value, value_by_file)
        elif variant == 'TOLOWER':
            value = args.pop(0)
            output_identifier = args.pop(0)

            output_value = value.lower()
            self.set_cmake_str_var(tline.file, output_identifier, output_value, output_value)
        elif variant == 'TOUPPER':
            value = args.pop(0)
            output_identifier = args.pop(0)

            output_value = value.upper()
            self.set_cmake_str_var(tline.file, output_identifier, output_value, output_value)
        elif variant == 'SUBSTRING':
            value = args.pop(0).encode()
            begin_value = int(args.pop(0))
            length_value = int(args.pop(0))
            output_identifier = args.pop(0)

            end_value = length_value if length_value < 0 else begin_value + length_value
            output_value = value[begin_value:end_value]
            output_value = output_value.decode()
            self.set_cmake_str_var(tline.file, output_identifier, output_value, output_value)
        elif variant == 'STRIP':
            value = args.pop(0)
            output_identifier = args.pop(0)

            output_value = value.strip(' ')
            self.set_cmake_str_var(tline.file, output_identifier, output_value, output_value)
        elif variant == 'COMPARE':
            comparison = args.pop(0)
            left_value = args.pop(0)
            right_value = args.pop(0)
            output_identifier = args.pop(0)

            output_value = None
            if comparison == 'LESS':
                output_value = left_value < right_value
            elif comparison == 'GREATER':
                output_value = left_value > right_value
            elif comparison == 'EQUAL':
                output_value = left_value == right_value
            elif comparison == 'NOTEQUAL':
                output_value = left_value != right_value
            elif comparison == 'LESS_EQUAL':
                output_value = left_value <= right_value
            elif comparison == 'GREATER_EQUAL':
                output_value = left_value >= right_value

            if output_value is not None:
                self.set_cmake_bool_var(tline.file, output_identifier, output_value, output_value)

    def _cmake_list(self, tline: CMakeTraceLine) -> None:
        """Handler for the CMake list() function in all variants.

        Comes in seventeen variants:
        list(LENGTH <list> <out-var>)
        list(GET <list> <element index> [<index> ...] <out-var>)
        list(JOIN <list> <glue> <out-var>)
        list(SUBLIST <list> <begin> <length> <out-var>)
        list(FIND <list> <value> <out-var>)
        list(APPEND <list> [<element>...])
        list(FILTER <list> {INCLUDE | EXCLUDE} REGEX <regex>)
        list(INSERT <list> <index> [<element>...])
        list(POP_BACK <list> [<out-var>...])
        list(POP_FRONT <list> [<out-var>...])
        list(PREPEND <list> [<element>...])
        list(REMOVE_ITEM <list> <value>...)
        list(REMOVE_AT <list> <index>...)
        list(REMOVE_DUPLICATES <list>)
        list(TRANSFORM <list> <ACTION> [...])
        list(REVERSE <list>)
        list(SORT <list> [...])

        We only support the following variants:
        - LENGTH
        - GET
        - JOIN
        - FIND
        - APPEND
        - INSERT
        - POP_BACK
        - POP_FRONT
        - PREPEND
        - REMOVE_ITEM
        - REMOVE_DUPLICATES
        Any uses of other variants will be ignored silently. The supported
        variants have some caveats:
        - We don't properly handle scoping, so calls to list() inside a
          function could incorrectly shadow the outer scope.
        """
        # DOC: https://cmake.org/cmake/help/latest/command/string.html

        args = list(tline.args)

        if len(args) < 1:
            return self._gen_exception('string', 'requires at least one argument', tline)

        variant = args.pop(0)
        if variant == 'LENGTH':
            identifier = args.pop(0)
            output_identifier = args.pop(0)

            value, value_by_file = self.get_cmake_list_var(tline.file, identifier, [])
            output_value = len(value)
            output_value_by_file = len(value_by_file)
            output_value = str(output_value)
            output_value_by_file = str(output_value_by_file)
            self.set_cmake_str_var(tline.file, output_identifier, output_value, output_value_by_file)
        elif variant == 'GET':
            identifier = args.pop(0)
            output_identifier = args.pop(-1)
            index_values = [int(index_value) for index_value in args]

            value, value_by_file = self.get_cmake_list_var(tline.file, identifier)
            if value is None:
                output_value = ['NOTFOUND']
            else:
                output_value = [value[index_value] for index_value in index_values]
            if value_by_file is None:
                output_value_by_file = ['NOTFOUND']
            else:
                output_value_by_file = [value_by_file[index_value] for index_value in index_values]
            self.set_cmake_list_var(tline.file, output_identifier, output_value, output_value_by_file)
        elif variant == 'JOIN':
            identifier = args.pop(0)
            glue_value = args.pop(0)
            output_identifier = args.pop(0)

            value, value_by_file = self.get_cmake_list_var(tline.file, identifier)
            output_value = 'NOTFOUND' if value is None else glue_value.join(value)
            output_value_by_file = 'NOTFOUND' if value_by_file is None else glue_value.join(value_by_file)
            self.set_cmake_str_var(tline.file, output_identifier, output_value, output_value_by_file)
        elif variant == 'FIND':
            identifier = args.pop(0)
            search_value = args.pop(0)
            output_identifier = args.pop(0)

            value, value_by_file = self.get_cmake_list_var(tline.file, identifier, [])
            try:
                output_value = value.index(search_value)
            except ValueError:
                output_value = -1
            try:
                output_value_by_file = value.index(search_value)
            except ValueError:
                output_value_by_file = -1
            output_value = [str(output_value)]
            output_value_by_file = [str(output_value_by_file)]
            self.set_cmake_list_var(tline.file, output_identifier, output_value, output_value_by_file)
        elif variant == 'APPEND':
            identifier = args.pop(0)
            if len(args) > 0:
                input_value = [v for arg in args for v in self._str_to_list(arg)]

                value, value_by_file = self.get_cmake_list_var(tline.file, identifier, [])
                value += input_value
                value_by_file += input_value
                self.set_cmake_list_var(tline.file, identifier, value, value_by_file)
        elif variant == 'INSERT':
            identifier = args.pop(0)
            if len(args) > 0:
                value = [v for arg in args for v in self._str_to_list(arg)]

                value, value_by_file = self.get_cmake_list_var(tline.file, identifier, [])
                value = args + value
                value_by_file = args + value_by_file
                self.set_cmake_list_var(tline.file, identifier, value, value_by_file)
        elif variant == 'POP_BACK':
            identifier = args.pop(0)
            output_identifiers = args

            value, value_by_file = self.get_cmake_list_var(tline.file, identifier, [])
            if len(output_identifiers) > 0:
                for output_identifier in output_identifiers:
                    try:
                        output_value = [value.pop(-1)]
                    except IndexError:
                        output_value = None
                    try:
                        output_value_by_file = [value_by_file.pop(-1)]
                    except IndexError:
                        output_value_by_file = None
                    self.set_cmake_list_var(tline.file, output_identifier, output_value, output_value_by_file)
            else:
                try:
                    value.pop(-1)
                except IndexError:
                    None
                try:
                    value_by_file.pop(-1)
                except IndexError:
                    None
        elif variant == 'POP_FRONT':
            identifier = args.pop(0)
            output_identifiers = args

            value, value_by_file = self.get_cmake_list_var(tline.file, identifier, [])
            if len(output_identifiers) > 0:
                for output_identifier in output_identifiers:
                    try:
                        output_value = [value.pop(0)]
                    except IndexError:
                        output_value = None
                    try:
                        output_value_by_file = [value_by_file.pop(0)]
                    except IndexError:
                        output_value_by_file = None
                    self.set_cmake_list_var(tline.file, output_identifier, output_value, output_value_by_file)
            else:
                try:
                    value.pop(0)
                except IndexError:
                    None
                try:
                    value_by_file.pop(0)
                except IndexError:
                    None
        elif variant == 'PREPEND':
            identifier = args.pop(0)
            if len(args) > 0:
                input_value = [v for arg in args for v in self._str_to_list(arg)]

                value, value_by_file = self.get_cmake_list_var(tline.file, identifier, [])
                value = input_value + value
                value_by_file = input_value + value_by_file
                self.set_cmake_list_var(tline.file, identifier, value, value_by_file)
        elif variant == 'REMOVE_ITEM':
            identifier = args.pop(0)
            search_values = args

            value, value_by_file = self.get_cmake_list_var(tline.file, identifier)
            if value is not None:
                value = list(filter(lambda v: v not in search_values, value))
            if value_by_file is not None:
                value_by_file = list(filter(lambda v: v not in search_values, value_by_file))
            self.set_cmake_list_var(tline.file, identifier, value, value_by_file)
        elif variant == 'REMOVE_DUPLICATES':
            identifier = args.pop(0)

            value, value_by_file = self.get_cmake_list_var(tline.file, identifier)
            if value is not None:
                value = list(dict.fromkeys(value).keys())
            if value_by_file is not None:
                value_by_file = list(dict.fromkeys(value_by_file).keys())
            self.set_cmake_list_var(tline.file, identifier, value, value_by_file)

    def _cmake_add_executable(self, tline: CMakeTraceLine) -> None:
        # DOC: https://cmake.org/cmake/help/latest/command/add_executable.html
        args = list(tline.args) # Make a working copy

        # Make sure the exe is imported
        is_imported = True
        if 'IMPORTED' not in args:
            return self._gen_exception('add_executable', 'non imported executables are not supported', tline)

        args.remove('IMPORTED')

        if len(args) < 1:
            return self._gen_exception('add_executable', 'requires at least 1 argument', tline)

        self.targets[args[0]] = CMakeTarget(args[0], 'EXECUTABLE', {}, tline=tline, imported=is_imported)

    def _cmake_add_library(self, tline: CMakeTraceLine) -> None:
        # DOC: https://cmake.org/cmake/help/latest/command/add_library.html
        args = list(tline.args) # Make a working copy

        # Make sure the lib is imported
        if 'INTERFACE' in args:
            args.remove('INTERFACE')

            if len(args) < 1:
                return self._gen_exception('add_library', 'interface library name not specified', tline)

            self.targets[args[0]] = CMakeTarget(args[0], 'INTERFACE', {}, tline=tline, imported='IMPORTED' in args)
        elif 'IMPORTED' in args:
            args.remove('IMPORTED')

            # Now, only look at the first two arguments (target_name and target_type) and ignore the rest
            if len(args) < 2:
                return self._gen_exception('add_library', 'requires at least 2 arguments', tline)

            self.targets[args[0]] = CMakeTarget(args[0], args[1], {}, tline=tline, imported=True)
        elif 'ALIAS' in args:
            args.remove('ALIAS')

            # Now, only look at the first two arguments (target_name and target_ref) and ignore the rest
            if len(args) < 2:
                return self._gen_exception('add_library', 'requires at least 2 arguments', tline)

            # Simulate the ALIAS with INTERFACE_LINK_LIBRARIES
            self.targets[args[0]] = CMakeTarget(args[0], 'ALIAS', {'INTERFACE_LINK_LIBRARIES': [args[1]]}, tline=tline)
        elif 'OBJECT' in args:
            return self._gen_exception('add_library', 'OBJECT libraries are not supported', tline)
        else:
            self.targets[args[0]] = CMakeTarget(args[0], 'NORMAL', {}, tline=tline)

    def _cmake_add_custom_command(self, tline: CMakeTraceLine, name: T.Optional[str] = None) -> None:
        # DOC: https://cmake.org/cmake/help/latest/command/add_custom_command.html
        args = self._flatten_args(list(tline.args))  # Commands can be passed as ';' separated lists

        if not args:
            return self._gen_exception('add_custom_command', 'requires at least 1 argument', tline)

        # Skip the second function signature
        if args[0] == 'TARGET':
            return self._gen_exception('add_custom_command', 'TARGET syntax is currently not supported', tline)

        magic_keys = ['OUTPUT', 'COMMAND', 'MAIN_DEPENDENCY', 'DEPENDS', 'BYPRODUCTS',
                      'IMPLICIT_DEPENDS', 'WORKING_DIRECTORY', 'COMMENT', 'DEPFILE',
                      'JOB_POOL', 'VERBATIM', 'APPEND', 'USES_TERMINAL', 'COMMAND_EXPAND_LISTS']

        target = CMakeGeneratorTarget(name)

        def handle_output(key: str, target: CMakeGeneratorTarget) -> None:
            target.outputs += [Path(key)]

        def handle_command(key: str, target: CMakeGeneratorTarget) -> None:
            if key == 'ARGS':
                return
            target.command[-1] += [key]

        def handle_depends(key: str, target: CMakeGeneratorTarget) -> None:
            target.depends += [key]

        working_dir = None

        def handle_working_dir(key: str, target: CMakeGeneratorTarget) -> None:
            nonlocal working_dir
            if working_dir is None:
                working_dir = key
            else:
                working_dir += ' '
                working_dir += key

        fn = None

        for i in args:
            if i in magic_keys:
                if i == 'OUTPUT':
                    fn = handle_output
                elif i == 'DEPENDS':
                    fn = handle_depends
                elif i == 'WORKING_DIRECTORY':
                    fn = handle_working_dir
                elif i == 'COMMAND':
                    fn = handle_command
                    target.command += [[]]
                else:
                    fn = None
                continue

            if fn is not None:
                fn(i, target)

        cbinary_dir, _ = self.get_cmake_str_var(None, 'MESON_PS_CMAKE_CURRENT_BINARY_DIR')
        csource_dir, _ = self.get_cmake_str_var(None, 'MESON_PS_CMAKE_CURRENT_SOURCE_DIR')

        target.working_dir     = Path(working_dir) if working_dir else None
        target.current_bin_dir = Path(cbinary_dir) if cbinary_dir else None
        target.current_src_dir = Path(csource_dir) if csource_dir else None
        target.outputs = [Path(x) for x in self._guess_files([str(y) for y in target.outputs])]
        target.depends = self._guess_files(target.depends)
        target.command = [self._guess_files(x) for x in target.command]

        self.custom_targets += [target]
        if name:
            self.targets[name] = target

    def _cmake_add_custom_target(self, tline: CMakeTraceLine) -> None:
        # DOC: https://cmake.org/cmake/help/latest/command/add_custom_target.html
        # We only the first parameter (the target name) is interesting
        if len(tline.args) < 1:
            return self._gen_exception('add_custom_target', 'requires at least one argument', tline)

        # It's pretty much the same as a custom command
        self._cmake_add_custom_command(tline, tline.args[0])

    def _cmake_set_property(self, tline: CMakeTraceLine) -> None:
        # DOC: https://cmake.org/cmake/help/latest/command/set_property.html
        args = list(tline.args)

        scope = args.pop(0)

        append = False
        append_string = False
        targets = []
        while args:
            curr = args.pop(0)
            if curr == 'APPEND':
                append = True
                continue
            if curr == 'APPEND_STRING':
                append_string = True
                continue

            if curr == 'PROPERTY':
                break

            targets += self._str_to_list(curr)

        if not args:
            return self._gen_exception('set_property', 'faild to parse argument list', tline)

        if len(args) == 1:
            # Tries to set property to nothing so nothing has to be done
            return

        identifier = args.pop(0)
        if self.trace_format == 'human':
            value = self._str_to_list(' '.join(args))
        else:
            value = [y for x in args for y in self._str_to_list(x)]
        if not value:
            return

        def do_target(t: str) -> None:
            if t not in self.targets:
                return self._gen_exception('set_property', f'TARGET {t} not found', tline)

            tgt = self.targets[t]
            if identifier not in tgt.properties:
                tgt.properties[identifier] = []

            tgt_properties = tgt.properties[identifier]
            if append_string:
                if len(tgt_properties) > 0:
                    tgt_properties[-1] += value.pop(0)
                tgt_properties += value
            elif append:
                tgt_properties += value
            else:
                tgt_properties = value

        def do_source(src: str) -> None:
            if identifier != 'HEADER_FILE_ONLY' or not self._list_to_bool(value):
                return

            current_src_dir, _ = self.get_cmake_str_var(None, 'MESON_PS_CMAKE_CURRENT_SOURCE_DIR')
            if not current_src_dir:
                mlog.warning(textwrap.dedent('''\
                    CMake trace: set_property(SOURCE) called before the preload script was loaded.
                    Unable to determine CMAKE_CURRENT_SOURCE_DIR. This can lead to build errors.
                '''))
                current_src_dir = '.'

            cur_p = Path(current_src_dir)
            src_p = Path(src)

            if not src_p.is_absolute():
                src_p = cur_p / src_p
            self.explicit_headers.add(src_p)

        if scope == 'TARGET':
            for i in targets:
                do_target(i)
        elif scope == 'SOURCE':
            files = self._guess_files(targets)
            for i in files:
                do_source(i)

    def _cmake_set_target_properties(self, tline: CMakeTraceLine) -> None:
        # DOC: https://cmake.org/cmake/help/latest/command/set_target_properties.html
        args = list(tline.args)

        targets = []
        while args:
            curr = args.pop(0)
            if curr == 'PROPERTIES':
                break

            targets.append(curr)

        # Now we need to try to reconsitute the original quoted format of the
        # arguments, as a property value could have spaces in it. Unlike
        # set_property() this is not context free. There are two approaches I
        # can think of, both have drawbacks:
        #
        #   1. Assume that the property will be capitalized ([A-Z_]), this is
        #      convention but cmake doesn't require it.
        #   2. Maintain a copy of the list here: https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html#target-properties
        #
        # Neither of these is awesome for obvious reasons. I'm going to try
        # option 1 first and fall back to 2, as 1 requires less code and less
        # synchroniztion for cmake changes.
        #
        # With the JSON output format, introduced in CMake 3.17, spaces are
        # handled properly and we don't have to do either options

        arglist = []  # type: T.List[T.Tuple[str, T.List[str]]]
        if self.trace_format == 'human':
            name = args.pop(0)
            values = []  # type: T.List[str]
            prop_regex = re.compile(r'^[A-Z_]+$')
            for a in args:
                if prop_regex.match(a):
                    if values:
                        arglist.append((name, self._str_to_list(' '.join(values))))
                    name = a
                    values = []
                else:
                    values.append(a)
            if values:
                arglist.append((name, self._str_to_list(' '.join(values))))
        else:
            arglist = [(x[0], self._str_to_list(x[1])) for x in zip(args[::2], args[1::2])]

        for name, value in arglist:
            for i in targets:
                if i not in self.targets:
                    return self._gen_exception('set_target_properties', f'TARGET {i} not found', tline)

                self.targets[i].properties[name] = value

    def _cmake_add_dependencies(self, tline: CMakeTraceLine) -> None:
        # DOC: https://cmake.org/cmake/help/latest/command/add_dependencies.html
        args = list(tline.args)

        if len(args) < 2:
            return self._gen_exception('add_dependencies', 'takes at least 2 arguments', tline)

        target = self.targets.get(args[0])
        if not target:
            return self._gen_exception('add_dependencies', 'target not found', tline)

        for i in args[1:]:
            target.depends += self._str_to_list(i)

    def _cmake_target_compile_definitions(self, tline: CMakeTraceLine) -> None:
        # DOC: https://cmake.org/cmake/help/latest/command/target_compile_definitions.html
        self._parse_common_target_options('target_compile_definitions', 'COMPILE_DEFINITIONS', 'INTERFACE_COMPILE_DEFINITIONS', tline)

    def _cmake_target_compile_options(self, tline: CMakeTraceLine) -> None:
        # DOC: https://cmake.org/cmake/help/latest/command/target_compile_options.html
        self._parse_common_target_options('target_compile_options', 'COMPILE_OPTIONS', 'INTERFACE_COMPILE_OPTIONS', tline)

    def _cmake_target_include_directories(self, tline: CMakeTraceLine) -> None:
        # DOC: https://cmake.org/cmake/help/latest/command/target_include_directories.html
        self._parse_common_target_options('target_include_directories', 'INCLUDE_DIRECTORIES', 'INTERFACE_INCLUDE_DIRECTORIES', tline, ignore=['SYSTEM', 'BEFORE'], paths=True)

    def _cmake_target_link_options(self, tline: CMakeTraceLine) -> None:
        # DOC: https://cmake.org/cmake/help/latest/command/target_link_options.html
        self._parse_common_target_options('target_link_options', 'LINK_OPTIONS', 'INTERFACE_LINK_OPTIONS', tline)

    def _cmake_target_link_libraries(self, tline: CMakeTraceLine) -> None:
        # DOC: https://cmake.org/cmake/help/latest/command/target_link_libraries.html
        # print('_cmake_target_link_libraries', tline)
        self._parse_common_target_options('target_link_options', 'LINK_LIBRARIES', 'INTERFACE_LINK_LIBRARIES', tline)

    def _parse_common_target_options(self, func: str, private_prop: str, interface_prop: str, tline: CMakeTraceLine, ignore: T.Optional[T.List[str]] = None, paths: bool = False) -> None:
        if ignore is None:
            ignore = ['BEFORE']

        args = list(tline.args)

        if len(args) < 1:
            return self._gen_exception(func, 'requires at least one argument', tline)

        target = args[0]
        if target not in self.targets:
            return self._gen_exception(func, f'TARGET {target} not found', tline)

        interface = []
        private = []

        mode = 'PUBLIC'
        for i in args[1:]:
            if i in ignore:
                continue

            if i in ['INTERFACE', 'LINK_INTERFACE_LIBRARIES', 'PUBLIC', 'PRIVATE', 'LINK_PUBLIC', 'LINK_PRIVATE']:
                mode = i
                continue

            if mode in ['INTERFACE', 'LINK_INTERFACE_LIBRARIES', 'PUBLIC', 'LINK_PUBLIC']:
                interface += self._str_to_list(i)

            if mode in ['PUBLIC', 'PRIVATE', 'LINK_PRIVATE']:
                private += self._str_to_list(i)

        if paths:
            interface = self._guess_files(interface)
            private = self._guess_files(private)

        interface = [x for x in interface if x]
        private = [x for x in private if x]

        for j in [(private_prop, private), (interface_prop, interface)]:
            if not j[0] in self.targets[target].properties:
                self.targets[target].properties[j[0]] = []

            self.targets[target].properties[j[0]] += j[1]

    def _meson_ps_execute_delayed_calls(self, tline: CMakeTraceLine) -> None:
        for l in self.stored_commands:
            fn = self.functions.get(l.func, None)
            if fn:
                fn(l)

        # clear the stored commands
        self.stored_commands = []

    def _meson_ps_reload_vars(self, tline: CMakeTraceLine) -> None:
        self.delayed_commands, _ = self.get_cmake_list_var(tline.file, 'MESON_PS_DELAYED_CALLS', [])

    def _meson_ps_disabled_function(self, tline: CMakeTraceLine) -> None:
        args = list(tline.args)
        if not args:
            mlog.error('Invalid preload.cmake script! At least one argument to `meson_ps_disabled_function` is expected')
            return
        mlog.warning(f'The CMake function "{args[0]}" was disabled to avoid compatibility issues with Meson.')

    def _lex_trace_human(self, trace: str) -> T.Generator[CMakeTraceLine, None, None]:
        # The trace format is: '<file>(<line>):  <func>(<args -- can contain \n> )\n'
        reg_tline = re.compile(r'\s*(.*\.(cmake|txt))\(([0-9]+)\):\s*(\w+)\(([\s\S]*?) ?\)\s*\n', re.MULTILINE)
        reg_other = re.compile(r'[^\n]*\n')
        loc = 0
        while loc < len(trace):
            mo_file_line = reg_tline.match(trace, loc)
            if not mo_file_line:
                skip_match = reg_other.match(trace, loc)
                if not skip_match:
                    print(trace[loc:])
                    raise CMakeException('Failed to parse CMake trace')

                loc = skip_match.end()
                continue

            loc = mo_file_line.end()

            file = mo_file_line.group(1)
            line = mo_file_line.group(3)
            func = mo_file_line.group(4)
            args = mo_file_line.group(5)
            argl = args.split(' ')
            argl = list(map(lambda x: x.strip(), argl))

            yield CMakeTraceLine(file, int(line), func, argl)

    def _lex_trace_json(self, trace: str) -> T.Generator[CMakeTraceLine, None, None]:
        lines = trace.splitlines(keepends=False)
        lines.pop(0)  # The first line is the version
        for i in lines:
            data = json.loads(i)
            assert isinstance(data['file'], str)
            assert isinstance(data['line'], int)
            assert isinstance(data['cmd'],  str)
            assert isinstance(data['args'], list)
            args = data['args']
            for j in args:
                assert isinstance(j, str)
            yield CMakeTraceLine(data['file'], data['line'], data['cmd'], args)

    def _flatten_args(self, args: T.List[str]) -> T.List[str]:
        # Split lists in arguments
        return [x for arg in args for x in self._str_to_list(arg)]

    def _guess_files(self, broken_list: T.List[str]) -> T.List[str]:
        # Nothing has to be done for newer formats
        if self.trace_format != 'human':
            return broken_list

        # Try joining file paths that contain spaces

        reg_start = re.compile(r'^([A-Za-z]:)?/(.*/)*[^./]+$')
        reg_end = re.compile(r'^.*\.[a-zA-Z]+$')

        fixed_list = []  # type: T.List[str]
        curr_str = None  # type: T.Optional[str]
        path_found = False # type: bool

        for i in broken_list:
            if curr_str is None:
                curr_str = i
                path_found = False
            elif Path(curr_str).is_file():
                # Abort concatenation if curr_str is an existing file
                fixed_list += [curr_str]
                curr_str = i
                path_found = False
            elif not reg_start.match(curr_str):
                # Abort concatenation if curr_str no longer matches the regex
                fixed_list += [curr_str]
                curr_str = i
                path_found = False
            elif reg_end.match(i):
                # File detected
                curr_str = f'{curr_str} {i}'
                fixed_list += [curr_str]
                curr_str = None
                path_found = False
            elif Path(f'{curr_str} {i}').exists():
                # Path detected
                curr_str = f'{curr_str} {i}'
                path_found = True
            elif path_found:
                # Add path to fixed_list after ensuring the whole path is in curr_str
                fixed_list += [curr_str]
                curr_str = i
                path_found = False
            else:
                curr_str = f'{curr_str} {i}'
                path_found = False

        if curr_str:
            fixed_list += [curr_str]
        return fixed_list
