# SPDX-License-Identifer: Apache-2.0
# Copyright 2020 The Meson development team
# Copyright © 2020-2021 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Meson specific typing helpers.

Holds typing helper classes, such as the ImmutableProtocol classes
"""

__all__ = [
    'ImmutableListProtocol',
    'ImmutableSetProtocol',
    'ParamSpec',
    'Protocol',
    'SizedSupportsStr',
    'SupportsStr',
]

import typing as _T

# We can change this to typing when we require Python 3.8
from typing_extensions import Protocol
# We can change this to typing when we require Python 3.10
from typing_extensions import ParamSpec


_T1 = _T.TypeVar('_T1')
# _T1_co = _T.TypeVar('_T1_co', covariant=True)
# _T1_contra = _T.TypeVar('_T1_contra', contravariant=True)
_T2 = _T.TypeVar('_T2')


class SupportsStr(Protocol):
    def __str__(self) -> str: ...


class SizedSupportsStr(_T.Sized, SupportsStr, Protocol):
    pass


class ImmutableListProtocol(Protocol[_T1]):
    """A protocol used in cases where a list is returned, but should not be
    mutated.

    This provides all of the methods of a Sequence, as well as copy(). copy()
    returns a list, which allows mutation as it's a copy and that's (hopefully)
    safe.

    One particular case this is important is for cached values, since python is
    a pass-by-reference language.
    """

    def __iter__(self) -> _T.Iterator[_T1]: ...

    @_T.overload
    def __getitem__(self, __index: int) -> _T1: ...
    @_T.overload
    def __getitem__(self, __index: slice) -> _T.List[_T1]: ...

    def __contains__(self, __item: object) -> bool: ...

    def __reversed__(self) -> _T.Iterator[_T1]: ...

    def __len__(self) -> int: ...

    def __add__(self, __other: _T.List[_T1]) -> _T.List[_T1]: ...

    def __eq__(self, __other: _T.Any) -> bool: ...
    def __ne__(self, __other: _T.Any) -> bool: ...
    def __le__(self, __other: _T.Any) -> bool: ...
    def __lt__(self, __other: _T.Any) -> bool: ...
    def __gt__(self, __other: _T.Any) -> bool: ...
    def __ge__(self, __other: _T.Any) -> bool: ...

    def count(self, __item: _T1) -> int: ...

    def index(self, __item: _T1) -> int: ...

    def copy(self) -> _T.List[_T1]: ...


class ImmutableSetProtocol(Protocol[_T1]):

    """A protocol for a set that cannot be mutated.

    This provides for cases where mutation of the set is undesired. Although
    this will be allowed at runtime, mypy (or another type checker), will see
    any attempt to use mutative methods as an error.
    """

    def __iter__(self) -> _T.Iterator[_T1]: ...

    def __contains__(self, __item: object) -> bool: ...

    def __len__(self) -> int: ...

    def __and__(self, __other: _T.AbstractSet[object]) -> _T.Set[_T1]: ...
    # def __or__(self, __other: _T.AbstractSet[_T1]) -> _T.Set[_T.Union[_T1, _T2]]: ...
    def __sub__(self, __other: _T.AbstractSet[_T.Union[_T1, None]]) -> _T.Set[_T1]: ...
    # def __xor__(self, __other: _T.AbstractSet[_T2]) -> _T.Set[_T.Union[_T1, _T2]]: ...

    def __eq__(self, __other: _T.Any) -> bool: ...
    def __ne__(self, __other: _T.Any) -> bool: ...
    def __le__(self, __other: _T.AbstractSet[object]) -> bool: ...
    def __lt__(self, __other: _T.AbstractSet[object]) -> bool: ...
    def __gt__(self, __other: _T.AbstractSet[object]) -> bool: ...
    def __ge__(self, __other: _T.AbstractSet[object]) -> bool: ...

    def copy(self) -> _T.Set[_T1]: ...

    def difference(self, *others: _T.Iterable[_T.Any]) -> _T.Set[_T1]: ...

    def intersection(self, *others: _T.Iterable[_T.Any]) -> _T.Set[_T1]: ...

    def isdisjoint(self, __other: _T.Iterable[_T.Any]) -> bool: ...

    def issubset(self, __other: _T.Iterable[_T.Any]) -> bool: ...

    def issuperset(self, __other: _T.Iterable[_T.Any]) -> bool: ...

    def symmetric_difference(self, __other: _T.Iterable[_T1]) -> _T.Set[_T1]: ...

    # def union(self, *others: _T.Iterable[_T2]) -> _T.Set[_T.Union[_T1, _T2]]: ...


def _main() -> None:
    """A helper function to ensure that type-checks succeed.
    """
    _0_0: SupportsStr = False
    _0 = _0_0,
    _1_0: SizedSupportsStr = [False]
    _1 = _1_0,
    _2_0: ImmutableListProtocol[bool] = [False]
    _2 = _2_0,
    _3_0: ImmutableSetProtocol[bool] = set((False,))
    # _3_1: ImmutableSetProtocol[bool] = frozenset((False,))
    _3 = _3_0, # _3_1
    _ = _0, _1, _2, _3

if __name__ == "__main__":
    _main()
