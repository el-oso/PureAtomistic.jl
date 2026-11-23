# The methods every functional defines, checked by TypeContracts. A concrete LDA or GGA also holds a
# `thresholds` field (see `Thresholds`); an `XC` holds its terms instead.

@contract AbstractFunctional "An exchange-correlation functional of the density, and for a GGA of σ." begin
    family(::Self)::Symbol => "`:lda` or `:gga`"
    kind(::Self)::Symbol => "`:exchange`, `:correlation` or `:exchange_correlation`"
    _with_number_type(::Self, ::Type) => "the functional with its parameters in another number type"
    _float32_max_order(::Type{Self})::Int => "largest derivative order evaluated in Float32"
end

@contract AbstractLDA "A functional of the density alone." begin
    _energy_density(::Self, ::Real) => "energy per volume of a restricted density, as a series"
    _energy_density(::Self, ::Tuple) => "energy per volume of a polarized density (ρα, ρβ)"
    _cutoff(::Self, ::Real) => "whether every output is zero at a restricted density"
    _cutoff(::Self, ::Tuple) => "whether every output is zero at a polarized density (ρα, ρβ)"
end

@contract AbstractGGA "A functional of the density and σ." begin
    _energy_density(::Self, ::Real, ::Real) => "energy per volume of a restricted (ρ, σ), as a series"
    _energy_density(::Self, ::Tuple, ::Tuple) => "energy per volume of a polarized (ρ, σ)"
    _cutoff(::Self, ::Real, ::Real) => "whether every output is zero at a restricted (ρ, σ)"
    _cutoff(::Self, ::Tuple, ::Tuple) => "whether every output is zero at a polarized (ρ, σ)"
end
