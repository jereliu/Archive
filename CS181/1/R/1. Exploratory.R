#### Caffè breva
require(magrittr)
require(dplyr)
require(plot3D)
require(xtable)

#### 1. Readin csv.gz. Data ####
dat <- read.csv("parameter.csv")
zMat <- matrix(dat$AME, nrow = length(15:24), 
               dimnames = list(unique(dat$depth), unique(dat$feature)))

xtable(zMat, digits = 6)

persp3D(x = 15:24, y = seq(14, 28, 2), z = zMat,
      theta = 30, phi = 30, 
      col = ramp.col(c("red", "orange", "yellow"), n = 100, alpha = 0.8), 
      border = "black",
      xlab = "Max Tree Depth", ylab = "Max Number of Feature",
      ticktype = "detailed")

#### 2. Forest Size ####
data.frame(B = c(10, 20, 50, 100, 500, 1000, 2000), 
           AME = c(0.272465381932, 0.272449839807, 
                   0.272440790648, 0.272434439889, 
                   0.272432452524
           ))


