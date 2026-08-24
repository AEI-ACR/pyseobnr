// Post-Newtonian truncation orders for the mode amplitude/phase coefficients
// (rho / f / delta). The coefficient arrays are always allocated to
// PN_limit_max; the number of orders actually summed is chosen per model at
// runtime via FluxParams.PN_limit: PN_limit_default for non-tidal models,
// PN_limit_max for tidal models (which carry higher-order tidal terms).
#define PN_limit_default 11 // PN orders summed for non-tidal models
#define PN_limit_max 16     // the coefficient-array allocation size
#define ell_max 8           // highest multipole index l