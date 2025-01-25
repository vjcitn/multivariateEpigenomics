
#' Use Robert Gregg's implementation of HALS for NMF in Julia
#' @import JuliaCall
#' @param x matrix
#' @param r target rank
#' @param alpha regularization factor, defaults to 0.2
#' @return a list with elements, X, W, H, res, call
#' @examples
#' mym = matrix(1:100,nr=10)*1.0
#' nn = nmf_HALS(mym, 4)
#' nn$res
#' rec = nn$W %*% t(nn$H)
#' rec[1:4,1:4]
#' @export
nmf_HALS = function(x, r, alpha=0.2, maxiter=250) {
  ca = match.call()
  tr = try(julia_setup())
  if (inherits(tr, "try-error")) stop("cannot start julia")
  tr =try(julia_source(system.file("julia/MethylationReduction.jl", package="multivariateEpigenomics")))
  if (inherits(tr, "try-error")) stop("could not source Methylation reduction code")
  tr =try(julia_source(system.file("julia/custom_nmf.jl", package="multivariateEpigenomics")))
  if (inherits(tr, "try-error")) stop("could not source custom nmf code")
  if (!(julia_exists("NMFCache"))) stop("NMFCache not found by julia_exists")
  julia_assign("X", x)
  cmd = sprintf("k=%d; nmf = NMFCache(X, k; α=%g);", as.integer(r), alpha)
  julia_command(cmd)
  cmd2 = sprintf("solveNMF(nmf, maxiter=%d);", as.integer(maxiter))
  julia_command(cmd2)
  X = julia_eval("nmf.X")
  W = julia_eval("nmf.W")
  H = julia_eval("nmf.H")
  julia_command("res = residual(nmf);")
  res = julia_eval("res")
  list(X=X, W=W, H=H, res=res, call=ca)
}

