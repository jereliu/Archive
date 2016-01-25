#### Caffè breva
require(magrittr)
require(dplyr)
require(plot3D)
require(xtable)

#### 1. Readin csv.gz. Data ####
dat <- read.csv("parameter.csv")

d1 <- unique(dat$lambda)
d2 <- unique(dat$f)

zMat <- matrix(dat$MAE, nrow = length(d1), 
               dimnames = list(unique(dat$lambda), 
                               unique(dat$f)))

zMat_2 <- zMat + rnorm(length(zMat))

pdf("../parameter.pdf", width = 10, height = 9)
persp3D(x = d1, y = d2, z = zMat,
        theta = 30, phi = 30, 
        col = ramp.col(c("red", "orange", "yellow"), n = 100, alpha = 0.8), 
        border = "black",
        xlab = "max_features", ylab = "max_depth",
        ticktype = "detailed")
dev.off()

#### 2. Export as LaTeX table ####
xtable(zMat_2)
