# Corrected version of the teammate's script.
# Changes from the original are marked ### FIX.
suppressPackageStartupMessages({library(readr); library(lme4)})

data_dir <- Sys.getenv("DATA_DIR", "data/raw")
c1_pheno <- read_csv(file.path(data_dir, "C1_Phenotype_Data_V2.csv"),
                     show_col_types = FALSE, progress = FALSE)

### FIX 1: shorthand_x is the POPULATION (499 of them), not the genotype (78,065
### lines). Using it means the model estimates population effects. Keep it, but name
### it honestly, and carry the real line id separately.
c1_dat <- data.frame(
  yield       = c1_pheno$YLD_BE,
  population  = factor(c1_pheno$shorthand_x),
  line        = factor(c1_pheno$LINE_UNIQUE_ID),
  environment = factor(paste(c1_pheno$LOC, c1_pheno$YEAR_x, sep = "_")),
  tester      = factor(c1_pheno$GERMPLASM_ID_TESTER)
)
c1_dat <- c1_dat[complete.cases(c1_dat[, c("yield","population","environment")]), ]
c1_dat <- droplevels(c1_dat)

cat("levels: population", nlevels(c1_dat$population),
    "| line", nlevels(c1_dat$line),
    "| environment", nlevels(c1_dat$environment),
    "| tester", nlevels(c1_dat$tester), "\n")

### FIX 2: every population uses exactly one tester, so tester is perfectly nested
### in population. Fitting both as fixed effects is rank-deficient and lmer silently
### drops columns. Drop tester.
tt <- tapply(c1_dat$tester, c1_dat$population, function(x) nlevels(droplevels(x)))
cat("max testers per population:", max(tt), "-> tester is nested, dropping it\n")

### FIX 3: the original called lmer() with no random-effects term, which errors.
### A model with only fixed effects is lm().
base <- lm(yield ~ environment + population, data = c1_dat)

### FIX 4: the original created `c1_fixedG_gxe` but then called anova() and update()
### on `c1_gxe`, which never exists.
### FIX 5: population:environment IS estimable (median 169 plots per cell), unlike
### line:environment, which has a single plot and is confounded with error.
gxe <- lmer(yield ~ environment + population + (1 | population:environment),
            data = c1_dat, REML = FALSE,
            control = lmerControl(optimizer = "bobyqa"))

### FIX 6: the original compared a REML fit to an ML fit. Likelihood-ratio tests
### require both on the same scale; use ML for the comparison.
base_ml <- lm(yield ~ environment + population, data = c1_dat)
print(anova(gxe, base_ml))

### FIX 7: refit with REML for unbiased variance components, then report them.
final <- update(gxe, REML = TRUE)
vc <- as.data.frame(VarCorr(final))
vc$pct <- 100 * vc$vcov / sum(vc$vcov)
print(vc[, c("grp","vcov","sdcor","pct")])
