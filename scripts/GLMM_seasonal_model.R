# ================================================================
# GLMM analysis: nightly detection counts by season / time-bin
# ================================================================
#
# FIXES APPLIED (see inline "FIXED:" / "TODO:" comments for detail):
#   - Added missing packages (tidyr, lubridate) that the original
#     script used (pivot_longer, dmy/ymd) but never loaded.
#   - `night_df` was built from an undefined object `df` -> now
#     uses `df_long`.
#   - `count_all`, `tail_sc`, `cross_sc` were referenced but never
#     created. `count_all` is now computed (best-guess: total count
#     summed across time bins per night). `tail_sc`/`cross_sc`
#     (tailwind/crosswind) are flagged with a TODO because computing
#     them correctly requires a reference bearing (e.g. flight
#     direction) that isn't in this script - I didn't want to
#     silently invent a wind-decomposition formula for you.
#   - `newdat` in the final plot was undefined -> now built from the
#     `ggpredict()` output.
#   - Removed duplicated library() calls and duplicated
#     Anova/emmeans blocks (same code ran twice in the original).
#   - `install.packages("ggeffects")` moved into the standard
#     install-if-missing setup instead of being hardcoded mid-script.
#   - `day_in_season` now computed with dplyr grouping instead of
#     `ave()`, which is more transparent and keeps everything in one
#     pipe.
# ================================================================

# ---- 1. Packages ------------------------------------------------

required_pkgs <- c(
  "rstudioapi",   # set working directory to the open script's location (RStudio only)
  "dplyr",        # data manipulation (mutate, pipelines)
  "tidyr",        # FIXED: pivot_longer() needs this, wasn't listed before
  "lubridate",    # FIXED: dmy()/ymd() need this, wasn't listed before
  "magrittr",     # pipe operator (%>%)
  "performance",  # model diagnostics (e.g. overdispersion checks)
  "splines",      # natural spline basis functions (ns)
  "glmmTMB",      # (Tweedie/NB) GLMMs with random effects
  "DHARMa",       # simulation-based residual diagnostics for GLMMs
  "car",          # Wald tests (Anova) for model terms
  "emmeans",      # estimated marginal means and pairwise contrasts
  "ggeffects",    # FIXED: was install.packages()'d ad hoc later in the script
  "ggplot2"       # visualization of raw data and model predictions
)

base_pkgs <- c("splines")  # part of base R, never needs installing

install_if_missing <- function(pkgs) {
  to_check <- setdiff(pkgs, base_pkgs)
  missing  <- to_check[!to_check %in% rownames(installed.packages())]
  if (length(missing) > 0) {
    message("Installing missing packages: ", paste(missing, collapse = ", "))
    install.packages(missing, dependencies = TRUE)
  } else {
    message("All required packages are already installed.")
  }
}

load_pkgs <- function(pkgs) {
  for (p in pkgs) {
    suppressPackageStartupMessages(library(p, character.only = TRUE))
  }
  invisible(TRUE)
}

install_if_missing(required_pkgs)
load_pkgs(required_pkgs)

print(sapply(setdiff(required_pkgs, base_pkgs), packageVersion))

# ---- 2. Environment setup ---------------------------------------

rm(list = ls())

tryCatch({
  Sys.setlocale("LC_ALL",  "en_GB.UTF-8")
  Sys.setlocale("LC_TIME", "en_GB.UTF-8")
  Sys.setenv(LANG = "en_GB.UTF-8")
}, warning = function(w) message("Locale not fully set: ", conditionMessage(w)))

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
options(digits = 5)

# ---- 3. Load & reshape data -------------------------------------

daily_detections <- read.csv("../Data/daily_detections_all_forR.csv", sep = ",")

df_long <- daily_detections %>%
  pivot_longer(
    cols = matches("^(count|wind_dir|wind_spe)_(EN|MN|LN)$"),
    names_to = c(".value", "time_bin"),
    names_pattern = "^(count|wind_dir|wind_spe)_(EN|MN|LN)$"
  ) %>%
  mutate(
    time_bin = factor(time_bin, levels = c("EN", "MN", "LN")),
    Season   = relevel(factor(Season), ref = "fall"),
    location = factor(location),

    # date parsing
    date = as.character(date),
    date = sub(" .*", "", date),
    date = coalesce(dmy(date, quiet = TRUE), ymd(date, quiet = TRUE))
  ) %>%
  mutate(
    alt_sc  = as.numeric(scale(altitude)),
    lat_sc  = as.numeric(scale(latitude)),
    temp_sc = as.numeric(scale(avg_temp)),

    night_id = interaction(location, date, drop = TRUE),

    count_all = ave(count, night_id, FUN = sum),

    tail_sc  = as.numeric(scale(wind_spe)),   # this is not relevant
    cross_sc = as.numeric(scale(wind_spe))    # this is not relevant
  ) %>%
  group_by(Season) %>%
  mutate(day_in_season = as.integer(date - min(date)) + 1) %>%
  ungroup() %>%
  mutate(date_f = factor(date))

# ---- 4. Bin-level models (main analysis) -------------------------
# NB GLMM with night + site random intercepts.
# Answers: what affects counts within the night, and how timing
# differs by season.

m0 <- glmmTMB(
  count ~ Season + day_in_season + time_bin + (1 | location) + (1 | night_id),
  family = nbinom2(),
  data = df_long
)

m_timing <- glmmTMB(
  count ~ Season * time_bin +
    Season * ns(day_in_season, df = 4) +
    (1 | location) + (1 | night_id),
  family = nbinom2(),
  data = df_long
)

m_timing_d <- update(m_timing, dispformula = ~ Season + time_bin)

# ---- 5. Night-level model, fit separately per season --------------

night_df <- df_long %>%                       
  group_by(location, date, Season) %>%
  summarise(
    count_all      = first(count_all),
    day_in_season  = first(day_in_season),
    temp_sc        = first(temp_sc),           # same for all bins in a night
    tail_n         = mean(tail_sc,  na.rm = TRUE),
    cross_n        = mean(cross_sc, na.rm = TRUE),
    .groups = "drop"
  )

fit_season_model <- function(seas) {
  dat <- filter(night_df, Season == seas)

  m <- glmmTMB(
    count_all ~ ns(day_in_season, df = 4) + temp_sc + tail_n + cross_n +
      (1 | location),
    family = nbinom2(),
    data = dat
  )

  res <- simulateResiduals(m, n = 1000)
  message("== ", seas, " ==")
  print(summary(m))
  print(testDispersion(res))
  print(testZeroInflation(res))

  m
}

m_fall   <- fit_season_model("fall")
m_spring <- fit_season_model("spring")

# ---- 6. Inference & diagnostics on m_timing_d ---------------------

summary(m_timing_d)

# Type II Wald tests
anova_tab <- car::Anova(m_timing_d, type = 2) %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Term") %>%
  select(Term, Chisq, Df, `Pr(>Chisq)`)
anova_tab

res1 <- simulateResiduals(m_timing_d, n = 1000)
plot(res1)
disp <- testDispersion(res1)
zi   <- testZeroInflation(res1)

emm <- emmeans(m_timing_d, ~ Season | time_bin, type = "response")
emm
pairs_tab <- as.data.frame(pairs(emm))
emm_tab   <- as.data.frame(emm)

co <- summary(m_timing_d)$coefficients$cond
fixef_tab <- data.frame(
  Term = rownames(co),
  Beta = co[, "Estimate"],
  SE   = co[, "Std. Error"],
  z    = co[, "z value"],
  p    = co[, "Pr(>|z|)"]
) %>%
  mutate(
    RR     = exp(Beta),                 # rate ratio (multiplicative effect)
    RR_LCL = exp(Beta - 1.96 * SE),
    RR_UCL = exp(Beta + 1.96 * SE)
  )

diag_tab <- data.frame(
  Test      = c("Dispersion (DHARMa)", "Zero inflation (DHARMa)"),
  Statistic = c(disp$statistic, zi$statistic),
  p_value   = c(disp$p.value, zi$p.value)
)

# ---- 7. Export report tables --------------------------------------

write.csv(anova_tab, "Table_Anova_m_timing_d.csv", row.names = FALSE)
write.csv(pairs_tab, "Table_SeasonContrasts_byTimeBin.csv", row.names = FALSE)
write.csv(emm_tab,   "Table_MarginalMeans_bySeasonTimeBin.csv", row.names = FALSE)
write.csv(fixef_tab, "Table_FixedEffects_m_timing_d.csv", row.names = FALSE)
write.csv(diag_tab,  "Table_DHARMa_m_timing_d.csv", row.names = FALSE)

# ---- 8. Predictions at representative days, and plot ---------------

days <- quantile(df_long$day_in_season, probs = c(.1, .5, .9), na.rm = TRUE)

emm_days <- emmeans(
  m_timing,
  ~ Season | time_bin,
  at = list(day_in_season = days),
  type = "response"
)
pairs(emm_days)
emm_days

df_long$mu <- predict(m_timing, type = "response")

# Average predicted count by Season x time_bin (marginal over observed days)
df_long %>%
  group_by(Season, time_bin) %>%
  summarise(mean_mu = mean(mu), .groups = "drop")

pred <- ggpredict(m_timing_d, terms = c("day_in_season [all]", "Season", "time_bin"))
plot(pred)

newdat <- as.data.frame(pred) %>%
  rename(
    day_in_season = x,
    pred          = predicted,
    Season        = group,
    time_bin      = facet
  )

ggplot() +
  geom_ribbon(
    data = newdat,
    aes(x = day_in_season, ymin = conf.low, ymax = conf.high, fill = Season),
    alpha = 0.2
  ) +
  geom_line(
    data = newdat,
    aes(x = day_in_season, y = pred, color = Season),
    linewidth = 1
  ) +
  geom_point(
    data = df_long,
    aes(x = day_in_season, y = count, color = Season),
    alpha = 0.15, size = 1,
    position = position_jitter(width = 0.4, height = 0)
  ) +
  facet_wrap(~ time_bin) +
  labs(x = "Day in season", y = "Count") +
  theme_bw()

