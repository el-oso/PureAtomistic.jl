# API

## Construction and read-out

```@docs
Taylor
variable
constant
value
derivatives
PureTaylor.coefficient
substitute
```

## Arithmetic and elementary functions

```@docs
Base.sqrt(::Taylor)
Base.cbrt(::Taylor)
Base.:^(::Taylor, ::Real)
PureTaylor.power_series
Base.:^(::Taylor, ::Taylor)
Base.exp(::Taylor)
Base.expm1(::Taylor)
Base.log(::Taylor)
Base.log1p(::Taylor)
```

## Fast functions

```@docs
fast_sqrt
fast_cbrt
fast_invcbrt
fast_log
fast_log1p
fast_exp
fast_expm1
```
