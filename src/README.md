# Third-party code

This folder contains copies of two third-party libraries, redistributed under their original licenses.

| Folder | Upstream | License |
|---|---|---|
| `tabpfn/` | [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN) | Apache License 2.0, see [`tabpfn/LICENSE`](tabpfn/LICENSE) |
| `tabicl/` | [soda-inria/tabicl](https://github.com/soda-inria/tabicl), commit [`02c52ad`](https://github.com/soda-inria/tabicl/tree/02c52ad) (after v2.0.3) | BSD 3-Clause, see [`tabicl/LICENSE`](tabicl/LICENSE). `tabicl/forecast/` is derived from [TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series) and is under the Apache License 2.0 (same file) |

**Modifications.** `tabicl/` is an unmodified copy of the upstream source at the commit above.
`tabpfn/` was modified for this project: in particular, its internal imports were rewritten to `src.tabpfn`
so that it can be used without installation.
