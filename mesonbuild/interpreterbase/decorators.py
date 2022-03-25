# Copyright 2013-2021 The Meson development team

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from .. import mesonlib, mlog, mparser
from .disabler import Disabler
from .exceptions import InterpreterException, InvalidArguments
from ._unholder import _unholder

from functools import wraps
import abc
import itertools
import copy
import typing as T

if T.TYPE_CHECKING:
    from typing_extensions import Protocol, TypedDict

    from ..modules import ModuleObject, ModuleState
    from .baseobjects import (
        InterpreterCallable, InterpreterCallableDecorator, InterpreterFunction, InterpreterMethod,
        InterpreterModuleMethod, InterpreterObject, InterpreterOperator, PS_rest, TV_func, TV_func_result,
        TV_interpreter_func, TV_interpreter_func_arg, TV_interpreter_func_args, TV_interpreter_func_args_2,
        TV_interpreter_func_args_co, TV_interpreter_func_kwargs, TV_interpreter_func_kwargs_2,
        TV_interpreter_func_kwargs_co, TV_interpreter_func_result, TV_interpreter_func_result_2,
        TV_interpreter_func_result_co, TV_interpreter_func_state, TV_interpreter_func_state_2,
        TV_interpreter_func_state_co, TV_interpreter_op_left, TV_interpreter_op_result, TV_interpreter_op_right,
        TYPE_var, TYPE_kwargs,
    )
    from .interpreterbase import InterpreterBase, SubProject
    from .operator import MesonOperator

def interpreter_func_decorator(decorator: T.Callable[
    [T.Callable[[TV_interpreter_func_state_co, TV_interpreter_func_args_co, TV_interpreter_func_kwargs_co], TV_interpreter_func_result_co]],
    T.Callable[[TV_interpreter_func_state, TV_interpreter_func_args, TV_interpreter_func_kwargs], TV_interpreter_func_result],
]) -> InterpreterCallableDecorator[
    TV_interpreter_func_state_co,
    TV_interpreter_func_args_co,
    TV_interpreter_func_kwargs_co,
    TV_interpreter_func_result_co,
    TV_interpreter_func_state,
    TV_interpreter_func_args,
    TV_interpreter_func_kwargs,
    TV_interpreter_func_result,
]:
    @T.overload
    def wrapped_decorator(
        interpreter_func: InterpreterFunction[TV_interpreter_func_state_co, TV_interpreter_func_args_co, TV_interpreter_func_kwargs_co, TV_interpreter_func_result_co],
    ) -> InterpreterFunction[TV_interpreter_func_state, TV_interpreter_func_args, TV_interpreter_func_kwargs, TV_interpreter_func_result]: ...

    @T.overload
    def wrapped_decorator(
        interpreter_func: InterpreterMethod[TV_interpreter_func_state_co, TV_interpreter_func_args_co, TV_interpreter_func_kwargs_co, TV_interpreter_func_result_co],
    ) -> InterpreterMethod[TV_interpreter_func_state, TV_interpreter_func_args, TV_interpreter_func_kwargs, TV_interpreter_func_result]: ...

    @T.overload
    def wrapped_decorator(
        interpreter_func: InterpreterModuleMethod[ModuleObject, TV_interpreter_func_state_co, TV_interpreter_func_args_co, TV_interpreter_func_kwargs_co, TV_interpreter_func_result_co],
    ) -> InterpreterModuleMethod[ModuleObject, TV_interpreter_func_state, TV_interpreter_func_args, TV_interpreter_func_kwargs, TV_interpreter_func_result]: ...

    @wraps(decorator)
    def wrapped_decorator(
        interpreter_func: InterpreterCallable[TV_interpreter_func_state_co, TV_interpreter_func_args_co, TV_interpreter_func_kwargs_co, TV_interpreter_func_result_co],
    ) -> InterpreterCallable[TV_interpreter_func_state, TV_interpreter_func_args, TV_interpreter_func_kwargs, TV_interpreter_func_result]:
        has_args = True
        storage = None

        # required to avoid contravariant type variables in function parameters
        def wrap_interpreter_func(
            inner_interpreter_func: InterpreterCallable[
                TV_interpreter_func_state_2, TV_interpreter_func_args_2, TV_interpreter_func_kwargs_2, TV_interpreter_func_result_2
            ]
        ) -> T.Callable[[
                TV_interpreter_func_state_2, TV_interpreter_func_args_2, TV_interpreter_func_kwargs_2
        ], TV_interpreter_func_result_2]:
            @wraps(inner_interpreter_func)
            def wrapped_interpreter_func(
                state: TV_interpreter_func_state_2, args: TV_interpreter_func_args_2, kwargs: TV_interpreter_func_kwargs_2
            ) -> TV_interpreter_func_result_2:
                nonlocal has_args, storage
                if storage is None:
                    if not has_args:
                        return inner_interpreter_func(state, args)
                    return T.cast(
                        'InterpreterMethod[TV_interpreter_func_state_2, TV_interpreter_func_args_2, TV_interpreter_func_kwargs_2, TV_interpreter_func_result_2]', inner_interpreter_func
                    )(state, args, kwargs)
                if isinstance(storage, mparser.FunctionNode):
                    return T.cast(
                        'InterpreterFunction[TV_interpreter_func_state_2, TV_interpreter_func_args_2, TV_interpreter_func_kwargs_2, TV_interpreter_func_result_2]', inner_interpreter_func
                    )(state, storage, args, kwargs)
                return T.cast(
                    'InterpreterModuleMethod[ModuleObject, TV_interpreter_func_state_2, TV_interpreter_func_args_2, TV_interpreter_func_kwargs_2, TV_interpreter_func_result_2]', inner_interpreter_func
                )(storage, state, args, kwargs)
            return wrapped_interpreter_func

        wrapped_interpreter_func = wrap_interpreter_func(interpreter_func)
        decorated_interpreter_func = decorator(wrapped_interpreter_func)

        @T.overload
        def wrapped_decorated_interpreter_func(__state: TV_interpreter_func_state, __node: mparser.FunctionNode, __args: TV_interpreter_func_args, __kwargs: TV_interpreter_func_kwargs) -> TV_interpreter_func_result: ...
        @T.overload
        def wrapped_decorated_interpreter_func(__state: TV_interpreter_func_state, __args: TV_interpreter_func_args, __kwargs: TV_interpreter_func_kwargs) -> TV_interpreter_func_result: ...
        @T.overload
        def wrapped_decorated_interpreter_func(__module_state: ModuleObject, __state: TV_interpreter_func_state, __args: TV_interpreter_func_args, __kwargs: TV_interpreter_func_kwargs) -> TV_interpreter_func_result: ...

        @wraps(interpreter_func)
        def wrapped_decorated_interpreter_func(*interpreter_func_args: T.Any) -> TV_interpreter_func_result:
            nonlocal has_args, storage
            state: TV_interpreter_func_state
            args: TV_interpreter_func_args
            kwargs: TV_interpreter_func_kwargs
            if len(interpreter_func_args) == 2:
                state, other = interpreter_func_args
                has_args = False
                return decorated_interpreter_func(state, other, None)
            if len(interpreter_func_args) == 3:
                state, args, kwargs = interpreter_func_args
            else:
                assert len(interpreter_func_args) == 4
                if isinstance(interpreter_func_args[1], mparser.FunctionNode):
                    node: mparser.FunctionNode
                    state, node, args, kwargs = interpreter_func_args
                    storage = node
                else:
                    module_state: ModuleObject
                    module_state, state, args, kwargs = interpreter_func_args
                    storage = module_state
            return decorated_interpreter_func(state, args, kwargs)
        return wrapped_decorated_interpreter_func
    return wrapped_decorator

@interpreter_func_decorator
def noPosargs(
    f: T.Callable[[TV_interpreter_func_state, TV_interpreter_func_args, TV_interpreter_func_kwargs], TV_interpreter_func_result]
) -> T.Callable[[TV_interpreter_func_state, TV_interpreter_func_args, TV_interpreter_func_kwargs], TV_interpreter_func_result]:
    @wraps(f)
    def wrapped(
        state: TV_interpreter_func_state, args: TV_interpreter_func_args, kwargs: TV_interpreter_func_kwargs
    ) -> TV_interpreter_func_result:
        if args:
            raise InvalidArguments('Function does not take positional arguments.')
        return f(state, args, kwargs)
    return wrapped

@interpreter_func_decorator
def noKwargs(
    f: T.Callable[[TV_interpreter_func_state, TV_interpreter_func_args, TV_interpreter_func_kwargs], TV_interpreter_func_result]
) -> T.Callable[[TV_interpreter_func_state, TV_interpreter_func_args, TV_interpreter_func_kwargs], TV_interpreter_func_result]:
    @wraps(f)
    def wrapped(
        state: TV_interpreter_func_state, args: TV_interpreter_func_args, kwargs: TV_interpreter_func_kwargs
    ) -> TV_interpreter_func_result:
        if kwargs:
            raise InvalidArguments('Function does not take keyword arguments.')
        return f(state, args, kwargs)
    return wrapped

@interpreter_func_decorator
def stringArgs(
    f: T.Callable[[TV_interpreter_func_state, T.List[str], TV_interpreter_func_kwargs], TV_interpreter_func_result]
) -> T.Callable[[TV_interpreter_func_state, T.List[str], TV_interpreter_func_kwargs], TV_interpreter_func_result]:
    @wraps(f)
    def wrapped(
        state: TV_interpreter_func_state, args: T.List[str], kwargs: TV_interpreter_func_kwargs
    ) -> TV_interpreter_func_result:
        if not isinstance(args, list):
            mlog.debug('Not a list:', str(args))
            raise InvalidArguments('Argument not a list.')
        if not all(isinstance(s, str) for s in args):
            mlog.debug('Element not a string:', str(args))
            raise InvalidArguments('Arguments must be strings.')
        return f(state, args, kwargs)
    return wrapped

def noArgsFlattening(f: TV_func) -> TV_func:
    setattr(f, 'no-args-flattening', True)  # noqa: B010
    return f

def noSecondLevelHolderResolving(f: TV_func) -> TV_func:
    setattr(f, 'no-second-level-holder-flattening', True)  # noqa: B010
    return f

def unholder_return(f: T.Callable[PS_rest, InterpreterObject]) -> T.Callable[PS_rest, TYPE_var]:
    @wraps(f)
    def wrapped(*meta_args: PS_rest.args, **meta_kwargs: PS_rest.kwargs) -> TYPE_var:
        res = f(*meta_args, **meta_kwargs)
        return _unholder(res)
    return wrapped

if T.TYPE_CHECKING:
    # https://github.com/python/mypy/issues/4617
    # DisablerDict = TypedDict('DisablerDict', {'disabler': bool}, total=False)

    class SupportsFound(Protocol):
        def found(self) -> bool: ...

    TV_disabler_dict = T.TypeVar('TV_disabler_dict', bound='T.Dict[str, object]')
    TV_supports_found = T.TypeVar('TV_supports_found', bound='SupportsFound')

@interpreter_func_decorator
def disablerIfNotFound(
    f: T.Callable[[TV_interpreter_func_state, TV_interpreter_func_args, TV_disabler_dict], TV_supports_found]
) -> T.Callable[[
    TV_interpreter_func_state, TV_interpreter_func_args, TV_disabler_dict
], T.Union[TV_supports_found, Disabler]]:
    @wraps(f)
    def wrapped(
        state: TV_interpreter_func_state, args: TV_interpreter_func_args, kwargs: TV_disabler_dict
    ) -> T.Union[TV_supports_found, Disabler]:
        disabler = kwargs.pop('disabler', False)
        ret = f(state, args, kwargs)
        if disabler and not ret.found():
            return Disabler()
        return ret
    return wrapped

def permittedKwargs(permitted: T.Set[str]) -> InterpreterCallableDecorator[
    TV_interpreter_func_state,
    TV_interpreter_func_args,
    TV_interpreter_func_kwargs,
    TV_interpreter_func_result,
    TV_interpreter_func_state,
    TV_interpreter_func_args,
    TV_interpreter_func_kwargs,
    TV_interpreter_func_result
]:
    @interpreter_func_decorator
    def inner(
        f: T.Callable[[TV_interpreter_func_state, TV_interpreter_func_args, TV_interpreter_func_kwargs], TV_interpreter_func_result]
    ) -> T.Callable[[TV_interpreter_func_state, TV_interpreter_func_args, TV_interpreter_func_kwargs], TV_interpreter_func_result]:
        @wraps(f)
        def wrapped(
            state: TV_interpreter_func_state, args: TV_interpreter_func_args, kwargs: TV_interpreter_func_kwargs
        ) -> TV_interpreter_func_result:
            unknowns = set(kwargs).difference(permitted)
            if unknowns:
                ustr = ', '.join([f'"{u}"' for u in sorted(unknowns)])
                raise InvalidArguments(f'Got unknown keyword arguments {ustr}')
            return f(state, args, kwargs)
        return wrapped
    return inner

def typed_operator(operator: MesonOperator, types: T.Union[T.Type, T.Tuple[T.Type, ...]]) -> T.Callable[[
    InterpreterOperator[TV_interpreter_op_left, TV_interpreter_op_right, TV_interpreter_op_result]
], InterpreterOperator[TV_interpreter_op_left, TV_interpreter_op_right, TV_interpreter_op_result]]:
    """Decorator that does type checking for operator calls.

    The principle here is similar to typed_pos_args, however much simpler
    since only one other object ever is passed
    """
    def inner(
        f: InterpreterOperator[TV_interpreter_op_left, TV_interpreter_op_right, TV_interpreter_op_result]
    ) -> InterpreterOperator[TV_interpreter_op_left, TV_interpreter_op_right, TV_interpreter_op_result]:
        @wraps(f)
        def wrapped(left: TV_interpreter_op_left, right: TV_interpreter_op_right) -> TV_interpreter_op_result:
            if not isinstance(right, types):
                raise InvalidArguments(f'The `{operator.value}` of {left.display_name()} does not accept objects of type {type(right).__name__} ({right})')
            return f(left, right)
        return wrapped
    return inner

def unary_operator(operator: MesonOperator) -> T.Callable[[
    InterpreterOperator[TV_interpreter_op_left, None, TV_interpreter_op_result]
], InterpreterOperator[TV_interpreter_op_left, None, TV_interpreter_op_result]]:
    """Decorator that does type checking for unary operator calls.

    This decorator is for unary operators that do not take any other objects.
    It should be impossible for a user to accidentally break this. Triggering
    this check always indicates a bug in the Meson interpreter.
    """
    def inner(
        f: InterpreterOperator[TV_interpreter_op_left, None, TV_interpreter_op_result]
    ) -> InterpreterOperator[TV_interpreter_op_left, None, TV_interpreter_op_result]:
        @wraps(f)
        def wrapped(left: TV_interpreter_op_left, right: None) -> TV_interpreter_op_result:
            if right is not None:
                raise mesonlib.MesonBugException(f'The unary operator `{operator.value}` of {left.display_name()} was passed the object {right} of type {type(right).__name__}')
            return f(left, right)
        return wrapped
    return inner

@T.overload
def typed_pos_args(
    __name: str,
    *types: T.Union[T.Type, T.Tuple[T.Type, ...]],
    varargs: None = None,
    optargs: None = None,
    min_varargs: int = ...,
    max_varargs: int = ...,
) -> InterpreterCallableDecorator[
    TV_interpreter_func_state,
    T.Tuple[TV_interpreter_func_arg, ...],
    TV_interpreter_func_kwargs,
    TV_interpreter_func_result,
    TV_interpreter_func_state,
    T.List[TV_interpreter_func_arg],
    TV_interpreter_func_kwargs,
    TV_interpreter_func_result
]: ...

@T.overload
def typed_pos_args(
    __name: str,
    *types: T.Union[T.Type, T.Tuple[T.Type, ...]],
    varargs: None = None,
    optargs: T.Optional[T.List[T.Union[T.Type, T.Tuple[T.Type, ...]]]] = None,
    min_varargs: int = ...,
    max_varargs: int = ...,
) -> InterpreterCallableDecorator[
    TV_interpreter_func_state,
    T.Tuple[T.Optional[TV_interpreter_func_arg], ...],
    TV_interpreter_func_kwargs,
    TV_interpreter_func_result,
    TV_interpreter_func_state,
    T.List[TV_interpreter_func_arg],
    TV_interpreter_func_kwargs,
    TV_interpreter_func_result
]: ...

@T.overload
def typed_pos_args(
    __name: str,
    varargs: T.Union[T.Type, T.Tuple[T.Type, ...]],
    optargs: None = None,
    min_varargs: int = ...,
    max_varargs: int = ...,
) -> InterpreterCallableDecorator[
    TV_interpreter_func_state,
    T.Tuple[T.List[TV_interpreter_func_arg]],
    TV_interpreter_func_kwargs,
    TV_interpreter_func_result,
    TV_interpreter_func_state,
    T.List[TV_interpreter_func_arg],
    TV_interpreter_func_kwargs,
    TV_interpreter_func_result
]: ...

@T.overload
def typed_pos_args(
    __name: str,
    *types: T.Union[T.Type, T.Tuple[T.Type, ...]],
    varargs: T.Optional[T.Union[T.Type, T.Tuple[T.Type, ...]]] = None,
    optargs: T.Optional[T.List[T.Union[T.Type, T.Tuple[T.Type, ...]]]] = None,
    min_varargs: int = 0,
    max_varargs: int = 0,
) -> InterpreterCallableDecorator[
    TV_interpreter_func_state,
    T.Tuple[T.Union[T.Optional[TV_interpreter_func_arg], T.List[TV_interpreter_func_arg]], ...],
    TV_interpreter_func_kwargs,
    TV_interpreter_func_result,
    TV_interpreter_func_state,
    T.List[TV_interpreter_func_arg],
    TV_interpreter_func_kwargs,
    TV_interpreter_func_result
]: ...

def typed_pos_args(
    __name: str,
    *types: T.Union[T.Type, T.Tuple[T.Type, ...]],
    varargs: T.Optional[T.Union[T.Type, T.Tuple[T.Type, ...]]] = None,
    optargs: T.Optional[T.List[T.Union[T.Type, T.Tuple[T.Type, ...]]]] = None,
    min_varargs: int = 0,
    max_varargs: int = 0,
) -> InterpreterCallableDecorator[
    TV_interpreter_func_state,
    T.Tuple[T.Union[T.Optional[TV_interpreter_func_arg], T.List[TV_interpreter_func_arg]], ...],
    TV_interpreter_func_kwargs,
    TV_interpreter_func_result,
    TV_interpreter_func_state,
    T.List[TV_interpreter_func_arg],
    TV_interpreter_func_kwargs,
    TV_interpreter_func_result
]:
    """Decorator that types type checking of positional arguments.

    This supports two different models of optional arguments, the first is the
    variadic argument model. Variadic arguments are a possibly bounded,
    possibly unbounded number of arguments of the same type (unions are
    supported). The second is the standard default value model, in this case
    a number of optional arguments may be provided, but they are still
    ordered, and they may have different types.

    This function does not support mixing variadic and default arguments.

    :__name: The name of the decorated function (as displayed in error messages)
    :varargs: They type(s) of any variadic arguments the function takes. If
        None the function takes no variadic args
    :min_varargs: the minimum number of variadic arguments taken
    :max_varargs: the maximum number of variadic arguments taken. 0 means unlimited
    :optargs: The types of any optional arguments parameters taken. If None
        then no optional parameters are taken.

    Some examples of usage blow:
    >>> @typed_pos_args('mod.func', str, (str, int))
    ... def func(self, state: ModuleState, args: T.Tuple[str, T.Union[str, int]], kwargs: T.Dict[str, T.Any]) -> T.Any:
    ...     pass

    >>> @typed_pos_args('method', str, varargs=str)
    ... def method(self, node: BaseNode, args: T.Tuple[str, T.List[str]], kwargs: T.Dict[str, T.Any]) -> T.Any:
    ...     pass

    >>> @typed_pos_args('method', varargs=str, min_varargs=1)
    ... def method(self, node: BaseNode, args: T.Tuple[T.List[str]], kwargs: T.Dict[str, T.Any]) -> T.Any:
    ...     pass

    >>> @typed_pos_args('method', str, optargs=[(str, int), str])
    ... def method(self, node: BaseNode, args: T.Tuple[str, T.Optional[T.Union[str, int]], T.Optional[str]], kwargs: T.Dict[str, T.Any]) -> T.Any:
    ...     pass

    When should you chose `typed_pos_args('name', varargs=str,
    min_varargs=1)` vs `typed_pos_args('name', str, varargs=str)`?

    The answer has to do with the semantics of the function, if all of the
    inputs are the same type (such as with `files()`) then the former is
    correct, all of the arguments are string names of files. If the first
    argument is something else the it should be separated.
    """
    @interpreter_func_decorator
    def inner(
        f: T.Callable[[
            TV_interpreter_func_state,
            T.Tuple[T.Union[T.Optional[TV_interpreter_func_arg], T.List[TV_interpreter_func_arg]], ...],
            TV_interpreter_func_kwargs
        ], TV_interpreter_func_result]
    ) -> T.Callable[[
        TV_interpreter_func_state, T.List[TV_interpreter_func_arg], TV_interpreter_func_kwargs
    ], TV_interpreter_func_result]:
        @wraps(f)
        def wrapped(
            state: TV_interpreter_func_state, args: T.List[TV_interpreter_func_arg], kwargs: TV_interpreter_func_kwargs
        ) -> TV_interpreter_func_result:
            # These are implementation programming errors, end users should never see them.
            assert isinstance(args, list), args
            assert max_varargs >= 0, 'max_varags cannot be negative'
            assert min_varargs >= 0, 'min_varags cannot be negative'
            assert optargs is None or varargs is None, \
                'varargs and optargs not supported together as this would be ambiguous'

            num_args = len(args)
            num_types = len(types)
            a_types = types

            if varargs:
                min_args = num_types + min_varargs
                max_args = num_types + max_varargs
                if max_varargs == 0 and num_args < min_args:
                    raise InvalidArguments(f'{__name} takes at least {min_args} arguments, but got {num_args}.')
                elif max_varargs != 0 and (num_args < min_args or num_args > max_args):
                    raise InvalidArguments(f'{__name} takes between {min_args} and {max_args} arguments, but got {num_args}.')
            elif optargs:
                if num_args < num_types:
                    raise InvalidArguments(f'{__name} takes at least {num_types} arguments, but got {num_args}.')
                elif num_args > num_types + len(optargs):
                    raise InvalidArguments(f'{__name} takes at most {num_types + len(optargs)} arguments, but got {num_args}.')
                # Add the number of positional arguments required
                if num_args > num_types:
                    diff = num_args - num_types
                    a_types = tuple(list(types) + list(optargs[:diff]))
            elif num_args != num_types:
                raise InvalidArguments(f'{__name} takes exactly {num_types} arguments, but got {num_args}.')

            for i, (arg, type_) in enumerate(itertools.zip_longest(args, a_types, fillvalue=varargs), start=1):
                if not isinstance(arg, type_):
                    if isinstance(type_, tuple):
                        shouldbe = 'one of: {}'.format(", ".join(f'"{t.__name__}"' for t in type_))
                    else:
                        shouldbe = f'"{type_.__name__}"'
                    raise InvalidArguments(f'{__name} argument {i} was of type "{type(arg).__name__}" but should have been {shouldbe}')

            # Ensure that we're actually passing a tuple.
            # Depending on what kind of function we're calling the length of
            # wrapped_args can vary.
            if varargs:
                # if we have varargs we need to split them into a separate
                # tuple, as python's typing doesn't understand tuples with
                # fixed elements and variadic elements, only one or the other.
                # so in that case we need T.Tuple[int, str, float, T.Tuple[str, ...]]
                args_tuple = tuple(itertools.chain(args[:len(types)], (args[len(types):],)))
                # print(f.__name__, 'args_tuple:', args_tuple, 'kwargs:', kwargs)
                return f(state, args_tuple, kwargs)
            if optargs and num_args < num_types + len(optargs):
                diff = num_types + len(optargs) - num_args
                args_tuple = tuple(itertools.chain(args, itertools.repeat(None, diff)))
                # print(f.__name__, 'args_tuple:', args_tuple, 'kwargs:', kwargs)
                return f(state, args_tuple, kwargs)
            args_tuple = tuple(args)
            # print(f.__name__, 'args_tuple:', args_tuple, 'kwargs:', kwargs)
            return f(state, args_tuple, kwargs)
        return wrapped
    return inner

class ContainerTypeInfo:

    """Container information for keyword arguments.

    For keyword arguments that are containers (list or dict), this class encodes
    that information.

    :param container: the type of container
    :param contains: the types the container holds
    :param pairs: if the container is supposed to be of even length.
        This is mainly used for interfaces that predate the addition of dictionaries, and use
        `[key, value, key2, value2]` format.
    :param allow_empty: Whether this container is allowed to be empty
        There are some cases where containers not only must be passed, but must
        not be empty, and other cases where an empty container is allowed.
    """

    def __init__(self, container: T.Type, contains: T.Union[T.Type, T.Tuple[T.Type, ...]], *,
                 pairs: bool = False, allow_empty: bool = True):
        self.container = container
        self.contains = contains
        self.pairs = pairs
        self.allow_empty = allow_empty

    def check(self, value: T.Any) -> bool:
        """Check that a value is valid.

        :param value: A value to check
        :return: True if it is valid, False otherwise
        """
        if not isinstance(value, self.container):
            return False
        iter_ = iter(value.values()) if isinstance(value, dict) else iter(value)
        for each in iter_:
            if not isinstance(each, self.contains):
                return False
        if self.pairs and len(value) % 2 != 0:
            return False
        if not value and not self.allow_empty:
            return False
        return True

    def description(self) -> str:
        """Human readable description of this container type.

        :return: string to be printed
        """
        container = 'dict' if self.container is dict else 'array'
        if isinstance(self.contains, tuple):
            contains = ' | '.join([t.__name__ for t in self.contains])
        else:
            contains = self.contains.__name__
        s = f'{container}[{contains}]'
        if self.pairs:
            s += ' that has even size'
        if not self.allow_empty:
            s += ' that cannot be empty'
        return s

_TV = T.TypeVar('_TV')

class _NULL_T:
    """Special null type for evolution, this is an implementation detail."""

_NULL = _NULL_T()

class KwargInfo(T.Generic[_TV]):

    """A description of a keyword argument to a meson function

    This is used to describe a value to the :func:typed_kwargs function.

    :param name: the name of the parameter
    :param types: A type or tuple of types that are allowed, or a :class:ContainerType
    :param required: Whether this is a required keyword argument. defaults to False
    :param listify: If true, then the argument will be listified before being
        checked. This is useful for cases where the Meson DSL allows a scalar or
        a container, but internally we only want to work with containers
    :param default: A default value to use if this isn't set. defaults to None,
        this may be safely set to a mutable type, as long as that type does not
        itself contain mutable types, typed_kwargs will copy the default
    :param since: Meson version in which this argument has been added. defaults to None
    :param since_message: An extra message to pass to FeatureNew when since is triggered
    :param deprecated: Meson version in which this argument has been deprecated. defaults to None
    :param deprecated_message: An extra message to pass to FeatureDeprecated
        when since is triggered
    :param validator: A callable that does additional validation. This is mainly
        intended for cases where a string is expected, but only a few specific
        values are accepted. Must return None if the input is valid, or a
        message if the input is invalid
    :param convertor: A callable that converts the raw input value into a
        different type. This is intended for cases such as the meson DSL using a
        string, but the implementation using an Enum. This should not do
        validation, just conversion.
    :param deprecated_values: a dictionary mapping a value to the version of
        meson it was deprecated in. The Value may be any valid value for this
        argument.
    :param since_values: a dictionary mapping a value to the version of meson it was
        added in.
    :param not_set_warning: A warning message that is logged if the kwarg is not
        set by the user.
    """
    def __init__(self, name: str,
                 types: T.Union[T.Type[_TV], T.Tuple[T.Union[T.Type[_TV], ContainerTypeInfo], ...], ContainerTypeInfo],
                 *, required: bool = False, listify: bool = False,
                 default: T.Optional[_TV] = None,
                 since: T.Optional[str] = None,
                 since_message: T.Optional[str] = None,
                 since_values: T.Optional[T.Dict[_TV, T.Union[str, T.Tuple[str, str]]]] = None,
                 deprecated: T.Optional[str] = None,
                 deprecated_message: T.Optional[str] = None,
                 deprecated_values: T.Optional[T.Dict[_TV, T.Union[str, T.Tuple[str, str]]]] = None,
                 validator: T.Optional[T.Callable[[T.Any], T.Optional[str]]] = None,
                 convertor: T.Optional[T.Callable[[_TV], object]] = None,
                 not_set_warning: T.Optional[str] = None):
        self.name = name
        self.types = types
        self.required = required
        self.listify = listify
        self.default = default
        self.since = since
        self.since_message = since_message
        self.since_values = since_values
        self.deprecated = deprecated
        self.deprecated_message = deprecated_message
        self.deprecated_values = deprecated_values
        self.validator = validator
        self.convertor = convertor
        self.not_set_warning = not_set_warning

    def evolve(self, *,
               name: T.Union[str, _NULL_T] = _NULL,
               required: T.Union[bool, _NULL_T] = _NULL,
               listify: T.Union[bool, _NULL_T] = _NULL,
               default: T.Union[_TV, None, _NULL_T] = _NULL,
               since: T.Union[str, None, _NULL_T] = _NULL,
               since_message: T.Union[str, None, _NULL_T] = _NULL,
               since_values: T.Union[T.Dict[_TV, T.Union[str, T.Tuple[str, str]]], None, _NULL_T] = _NULL,
               deprecated: T.Union[str, None, _NULL_T] = _NULL,
               deprecated_message: T.Union[str, None, _NULL_T] = _NULL,
               deprecated_values: T.Union[T.Dict[_TV, T.Union[str, T.Tuple[str, str]]], None, _NULL_T] = _NULL,
               validator: T.Union[T.Callable[[_TV], T.Optional[str]], None, _NULL_T] = _NULL,
               convertor: T.Union[T.Callable[[_TV], TYPE_var], None, _NULL_T] = _NULL) -> 'KwargInfo':
        """Create a shallow copy of this KwargInfo, with modifications.

        This allows us to create a new copy of a KwargInfo with modifications.
        This allows us to use a shared kwarg that implements complex logic, but
        has slight differences in usage, such as being added to different
        functions in different versions of Meson.

        The use the _NULL special value here allows us to pass None, which has
        meaning in many of these cases. _NULL itself is never stored, always
        being replaced by either the copy in self, or the provided new version.
        """
        return type(self)(
            name if not isinstance(name, _NULL_T) else self.name,
            self.types,
            listify=listify if not isinstance(listify, _NULL_T) else self.listify,
            required=required if not isinstance(required, _NULL_T) else self.required,
            default=default if not isinstance(default, _NULL_T) else self.default,
            since=since if not isinstance(since, _NULL_T) else self.since,
            since_message=since_message if not isinstance(since_message, _NULL_T) else self.since_message,
            since_values=since_values if not isinstance(since_values, _NULL_T) else self.since_values,
            deprecated=deprecated if not isinstance(deprecated, _NULL_T) else self.deprecated,
            deprecated_message=deprecated_message if not isinstance(deprecated_message, _NULL_T) else self.deprecated_message,
            deprecated_values=deprecated_values if not isinstance(deprecated_values, _NULL_T) else self.deprecated_values,
            validator=validator if not isinstance(validator, _NULL_T) else self.validator,
            convertor=convertor if not isinstance(convertor, _NULL_T) else self.convertor,
        )

_TV_typed_kwargs_dict = T.TypeVar('_TV_typed_kwargs_dict', bound='T.Dict[str, object]')

def typed_kwargs(name: str, *types: KwargInfo) -> InterpreterCallableDecorator[
    TV_interpreter_func_state,
    TV_interpreter_func_args,
    _TV_typed_kwargs_dict,
    TV_interpreter_func_result,
    TV_interpreter_func_state,
    TV_interpreter_func_args,
    _TV_typed_kwargs_dict,
    TV_interpreter_func_result
]:
    """Decorator for type checking keyword arguments.

    Used to wrap a meson DSL implementation function, where it checks various
    things about keyword arguments, including the type, and various other
    information. For non-required values it sets the value to a default, which
    means the value will always be provided.

    If type tyhpe is a :class:ContainerTypeInfo, then the default value will be
    passed as an argument to the container initializer, making a shallow copy

    :param name: the name of the function, including the object it's attached to
        (if applicable)
    :param *types: KwargInfo entries for each keyword argument.
    """
    @interpreter_func_decorator
    def inner(
        f: T.Callable[[TV_interpreter_func_state, TV_interpreter_func_args, _TV_typed_kwargs_dict], TV_interpreter_func_result]
    ) -> T.Callable[[TV_interpreter_func_state, TV_interpreter_func_args, _TV_typed_kwargs_dict], TV_interpreter_func_result]:
        def types_description(types_tuple: T.Tuple[T.Union[T.Type, ContainerTypeInfo], ...]) -> str:
            candidates = []
            for t in types_tuple:
                if isinstance(t, ContainerTypeInfo):
                    candidates.append(t.description())
                else:
                    candidates.append(t.__name__)
            shouldbe = 'one of: ' if len(candidates) > 1 else ''
            shouldbe += ', '.join(candidates)
            return shouldbe

        def raw_description(t: object) -> str:
            """describe a raw type (ie, one that is not a ContainerTypeInfo)."""
            if isinstance(t, list):
                if t:
                    return f"array[{' | '.join(sorted(mesonlib.OrderedSet(type(v).__name__ for v in t)))}]"
                return 'array[]'
            elif isinstance(t, dict):
                if t:
                    return f"dict[{' | '.join(sorted(mesonlib.OrderedSet(type(v).__name__ for v in t.values())))}]"
                return 'dict[]'
            return type(t).__name__

        def check_value_type(types_tuple: T.Tuple[T.Union[T.Type, ContainerTypeInfo], ...],
                             value: T.Any) -> bool:
            for t in types_tuple:
                if isinstance(t, ContainerTypeInfo):
                    if t.check(value):
                        return True
                elif isinstance(value, t):
                    return True
            return False

        @wraps(f)
        def wrapped(
            state: TV_interpreter_func_state, args: TV_interpreter_func_args, kwargs: _TV_typed_kwargs_dict
        ) -> TV_interpreter_func_result:
            def emit_feature_change(values: T.Dict[str, T.Union[str, T.Tuple[str, str]]], feature: T.Union[T.Type['FeatureDeprecated'], T.Type['FeatureNew']]) -> None:
                for n, version in values.items():
                    if isinstance(value, (dict, list)):
                        warn = n in value
                    else:
                        warn = n == value

                    if warn:
                        if isinstance(version, tuple):
                            version, msg = version
                        else:
                            msg = None
                        feature.single_use(f'"{name}" keyword argument "{info.name}" value "{n}"', version, state.subproject, msg, location=state.current_node)

            all_names = {t.name for t in types}
            unknowns = set(kwargs).difference(all_names)
            if unknowns:
                ustr = ', '.join([f'"{u}"' for u in sorted(unknowns)])
                raise InvalidArguments(f'{name} got unknown keyword arguments {ustr}')

            for info in types:
                types_tuple = info.types if isinstance(info.types, tuple) else (info.types,)
                value = kwargs.get(info.name)
                if value is not None:
                    if info.since:
                        feature_name = info.name + ' arg in ' + name
                        FeatureNew.single_use(feature_name, info.since, state.subproject, info.since_message, location=state.current_node)
                    if info.deprecated:
                        feature_name = info.name + ' arg in ' + name
                        FeatureDeprecated.single_use(feature_name, info.deprecated, state.subproject, info.deprecated_message, location=state.current_node)
                    if info.listify:
                        kwargs[info.name] = value = mesonlib.listify(value)
                    if not check_value_type(types_tuple, value):
                        shouldbe = types_description(types_tuple)
                        raise InvalidArguments(f'{name} keyword argument {info.name!r} was of type {raw_description(value)} but should have been {shouldbe}')

                    if info.validator is not None:
                        msg = info.validator(value)
                        if msg is not None:
                            raise InvalidArguments(f'{name} keyword argument "{info.name}" {msg}')

                    if info.deprecated_values is not None:
                        emit_feature_change(info.deprecated_values, FeatureDeprecated)

                    if info.since_values is not None:
                        emit_feature_change(info.since_values, FeatureNew)

                elif info.required:
                    raise InvalidArguments(f'{name} is missing required keyword argument "{info.name}"')
                else:
                    # set the value to the default, this ensuring all kwargs are present
                    # This both simplifies the typing checking and the usage
                    assert check_value_type(types_tuple, info.default), f'In funcion {name} default value of {info.name} is not a valid type, got {type(info.default)} expected {types_description(types_tuple)}'
                    # Create a shallow copy of the container. This allows mutable
                    # types to be used safely as default values
                    kwargs[info.name] = copy.copy(info.default)
                    if info.not_set_warning:
                        mlog.warning(info.not_set_warning)

                if info.convertor:
                    kwargs[info.name] = info.convertor(kwargs[info.name])

            return f(state, args, kwargs)
        return wrapped
    return inner

# This cannot be a dataclass due to https://github.com/python/mypy/issues/5374
class FeatureCheckBase(metaclass=abc.ABCMeta):
    "Base class for feature version checks"

    feature_registry: T.ClassVar[T.Dict[str, T.Dict[str, T.Set[T.Tuple[str, T.Optional['mparser.BaseNode']]]]]]
    emit_notice = False

    def __init__(self, feature_name: str, feature_version: str, extra_message: str = ''):
        self.feature_name = feature_name  # type: str
        self.feature_version = feature_version    # type: str
        self.extra_message = extra_message  # type: str

    @staticmethod
    def get_target_version(subproject: str) -> str:
        # Don't do any checks if project() has not been parsed yet
        if subproject not in mesonlib.project_meson_versions:
            return ''
        return mesonlib.project_meson_versions[subproject]

    @staticmethod
    @abc.abstractmethod
    def check_version(target_version: str, feature_version: str) -> bool:
        pass

    def use(self, subproject: 'SubProject', location: T.Optional['mparser.BaseNode'] = None) -> None:
        tv = self.get_target_version(subproject)
        # No target version
        if tv == '':
            return
        # Target version is new enough, don't warn
        if self.check_version(tv, self.feature_version) and not self.emit_notice:
            return
        # Feature is too new for target version or we want to emit notices, register it
        if subproject not in self.feature_registry:
            self.feature_registry[subproject] = {self.feature_version: set()}
        register = self.feature_registry[subproject]
        if self.feature_version not in register:
            register[self.feature_version] = set()

        feature_key = (self.feature_name, location)
        if feature_key in register[self.feature_version]:
            # Don't warn about the same feature multiple times
            # FIXME: This is needed to prevent duplicate warnings, but also
            # means we won't warn about a feature used in multiple places.
            return
        register[self.feature_version].add(feature_key)
        # Target version is new enough, don't warn even if it is registered for notice
        if self.check_version(tv, self.feature_version):
            return
        self.log_usage_warning(tv, location)

    @classmethod
    def report(cls, subproject: str) -> None:
        if subproject not in cls.feature_registry:
            return
        warning_str = cls.get_warning_str_prefix(cls.get_target_version(subproject))
        notice_str = cls.get_notice_str_prefix(cls.get_target_version(subproject))
        fv = cls.feature_registry[subproject]
        tv = cls.get_target_version(subproject)
        for version in sorted(fv.keys()):
            if cls.check_version(tv, version):
                notice_str += '\n * {}: {}'.format(version, {i[0] for i in fv[version]})
            else:
                warning_str += '\n * {}: {}'.format(version, {i[0] for i in fv[version]})
        if '\n' in notice_str:
            mlog.notice(notice_str, fatal=False)
        if '\n' in warning_str:
            mlog.warning(warning_str)

    def log_usage_warning(self, tv: str, location: T.Optional['mparser.BaseNode']) -> None:
        raise InterpreterException('log_usage_warning not implemented')

    @staticmethod
    def get_warning_str_prefix(tv: str) -> str:
        raise InterpreterException('get_warning_str_prefix not implemented')

    @staticmethod
    def get_notice_str_prefix(tv: str) -> str:
        raise InterpreterException('get_notice_str_prefix not implemented')

    def __call__(self, f: TV_interpreter_func) -> TV_interpreter_func:
        @interpreter_func_decorator
        def inner(
            inner_f: T.Callable[[
                TV_interpreter_func_state, TV_interpreter_func_args, TV_interpreter_func_kwargs
            ], TV_interpreter_func_result]
        ) -> T.Callable[[
            TV_interpreter_func_state, TV_interpreter_func_args, TV_interpreter_func_kwargs
        ], TV_interpreter_func_result]:
            @wraps(inner_f)
            def wrapped(
                state: TV_interpreter_func_state, args: TV_interpreter_func_args, kwargs: TV_interpreter_func_kwargs
            ) -> TV_interpreter_func_result:
                if state.subproject is None:
                    raise AssertionError(f'{state!r}')
                self.use(state.subproject, getattr(state, 'current_node', None))
                return inner_f(state, args, kwargs)
            return wrapped
        return inner(f)

    @classmethod
    def single_use(cls, feature_name: str, version: str, subproject: 'SubProject',
                   extra_message: T.Optional[str] = None, location: T.Optional['mparser.BaseNode'] = None) -> None:
        """Oneline version that instantiates and calls use()."""
        cls(feature_name, version, '' if extra_message is None else extra_message).use(subproject, location)

class FeatureNew(FeatureCheckBase):
    """Checks for new features"""

    # Class variable, shared across all instances
    #
    # Format: {subproject: {feature_version: set(feature_names)}}
    feature_registry = {}  # type: T.ClassVar[T.Dict[str, T.Dict[str, T.Set[T.Tuple[str, T.Optional[mparser.BaseNode]]]]]]

    @staticmethod
    def check_version(target_version: str, feature_version: str) -> bool:
        return mesonlib.version_compare_condition_with_min(target_version, feature_version)

    @staticmethod
    def get_warning_str_prefix(tv: str) -> str:
        return f'Project specifies a minimum meson_version \'{tv}\' but uses features which were added in newer versions:'

    @staticmethod
    def get_notice_str_prefix(tv: str) -> str:
        return ''

    def log_usage_warning(self, tv: str, location: T.Optional['mparser.BaseNode']) -> None:
        args = [
            'Project targeting', f"'{tv}'",
            'but tried to use feature introduced in',
            f"'{self.feature_version}':",
            f'{self.feature_name}.',
        ]
        if self.extra_message:
            args.append(self.extra_message)
        mlog.warning(*args, location=location)

class FeatureDeprecated(FeatureCheckBase):
    """Checks for deprecated features"""

    # Class variable, shared across all instances
    #
    # Format: {subproject: {feature_version: set(feature_names)}}
    feature_registry = {}  # type: T.ClassVar[T.Dict[str, T.Dict[str, T.Set[T.Tuple[str, T.Optional[mparser.BaseNode]]]]]]
    emit_notice = True

    @staticmethod
    def check_version(target_version: str, feature_version: str) -> bool:
        # For deprecation checks we need to return the inverse of FeatureNew checks
        return not mesonlib.version_compare_condition_with_min(target_version, feature_version)

    @staticmethod
    def get_warning_str_prefix(tv: str) -> str:
        return 'Deprecated features used:'

    @staticmethod
    def get_notice_str_prefix(tv: str) -> str:
        return 'Future-deprecated features used:'

    def log_usage_warning(self, tv: str, location: T.Optional['mparser.BaseNode']) -> None:
        args = [
            'Project targeting', f"'{tv}'",
            'but tried to use feature deprecated since',
            f"'{self.feature_version}':",
            f'{self.feature_name}.',
        ]
        if self.extra_message:
            args.append(self.extra_message)
        mlog.warning(*args, location=location)

# This cannot be a dataclass due to https://github.com/python/mypy/issues/5374
class FeatureCheckKwargsBase(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def feature_check_class(self) -> T.Type[FeatureCheckBase]:
        pass

    def __init__(self, feature_name: str, feature_version: str,
                 kwargs: T.List[str], extra_message: T.Optional[str] = None):
        self.feature_name = feature_name
        self.feature_version = feature_version
        self.kwargs = kwargs
        self.extra_message = extra_message

    def __call__(self, f: TV_interpreter_func) -> TV_interpreter_func:
        @interpreter_func_decorator
        def inner(
            inner_f: T.Callable[[
                TV_interpreter_func_state, TV_interpreter_func_args, TV_interpreter_func_kwargs
            ], TV_interpreter_func_result]
        ) -> T.Callable[[
            TV_interpreter_func_state, TV_interpreter_func_args, TV_interpreter_func_kwargs
        ], TV_interpreter_func_result]:
            @wraps(inner_f)
            def wrapped(
                state: TV_interpreter_func_state, args: TV_interpreter_func_args, kwargs: TV_interpreter_func_kwargs
            ) -> TV_interpreter_func_result:
                if state.subproject is None:
                    raise AssertionError(f'{state!r}')
                for arg in self.kwargs:
                    if arg not in kwargs:
                        continue
                    name = arg + ' arg in ' + self.feature_name
                    self.feature_check_class.single_use(
                            name, self.feature_version, state.subproject, self.extra_message, getattr(state, 'current_node', None))
                return inner_f(state, args, kwargs)
            return wrapped
        return inner(f)

class FeatureNewKwargs(FeatureCheckKwargsBase):
    feature_check_class = FeatureNew

class FeatureDeprecatedKwargs(FeatureCheckKwargsBase):
    feature_check_class = FeatureDeprecated
