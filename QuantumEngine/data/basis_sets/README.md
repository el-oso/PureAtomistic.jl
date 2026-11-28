# Basis set data

The 72 `.gbs` files in this directory are Gaussian94-format exports of the
[Basis Set Exchange](https://www.basissetexchange.org) (BSE), version 0.12, as
each file's header states (`! Basis Set Exchange`, `! Version 0.12`). The
download date is unknown: the files do not record it. The files are kept byte
for byte as exported and are never edited.

The selection of basis sets is the one the Rust `quantum_engine` program ships
in its own `basis_sets/` directory, of which QuantumEngine is a rewrite; the
files here are copies of those files.

## Citation

The Basis Set Exchange asks that work using it cite:

> B. P. Pritchard, D. Altarawy, B. Didier, T. D. Gibson and T. L. Windus,
> "New Basis Set Exchange: An Open, Up-to-Date Resource for the Molecular
> Sciences Community", *J. Chem. Inf. Model.* **59**(11), 4814–4820 (2019),
> DOI [10.1021/acs.jcim.9b00725](https://doi.org/10.1021/acs.jcim.9b00725).

Each file's header cites the original authors of its basis set; cite them as
well.

## License

No license or terms of use specific to the data were found, as read on
2026-11-28: not in the `MolSSI-BSE/basis_set_exchange` repository (commit
`4adaf1372c7101620ca1a9f3130be9ae97fb8f30`), whose data directory
`basis_set_exchange/data/` has no README, LICENSE or NOTICE file and whose
notes files give provenance only; not in its documentation, which says "The
code is released under the BSD 3-Clause license"; and not on the website, which
has a copyright line ("© 2020-2026 The Molecular Sciences Software Institute
(MolSSI), Virginia Tech") and no terms page. The one license is the repository's
BSD 3-Clause license, which covers the `basis_set_exchange` package; the data
is distributed inside that package.

The data is therefore shipped here as covered by that license. The project
owner accepted these terms on 2026-11-28. The license, as it reads at the
commit above:

```
Copyright (c) 2020 The Molecular Sciences Software Institute, Virginia Tech

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## Files not shipped

Four files of the Rust program's `basis_sets/` directory are not shipped:

- `def2-svp-jkfit.gbs`: a placeholder holding H, C, N and O only. The name
  `def2-SVP-JKFIT` resolves to the shipped `def2-universal-jkfit.gbs`.
- `6-31g_dp.gbs`, `6-311plusplusg_dp.gbs` and `6-311++g_2d2p.gbs`: they have no
  Basis Set Exchange header, and their source is not identified. The names
  `6-31G**`, `6-311++G**` and `6-311++G(2d,2p)` resolve to the BSE files
  `6-31g_star_star.gbs`, `6-311plusplusg_star_star.gbs` and
  `6-311plusplusg(2d,2p).gbs`. Two of the three differ from the BSE files of
  the same basis: in `6-31g_dp.gbs` the O d exponent is 1.285 against 0.8, and
  in `6-311plusplusg_dp.gbs` C, N, O, F and Ne have 9 shells against 10. The H,
  C and N blocks of `6-311++g_2d2p.gbs`, the only elements it has, are
  identical to those of the BSE file. The three files were not checked against
  the original papers (Hariharan and Pople 1973; Krishnan, Binkley, Seeger and
  Pople 1980; Frisch, Pople and Binkley 1984), which could not be reached.
