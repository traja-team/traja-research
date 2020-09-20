### Script to run the analysis presented in "Analysis of temporal patterns in animal movement networks". 
#Cristian Pasquaretta, Thibault Dubois, Tamara Gomez-Moracho, Virginie Perilhon Delepoulle, Guillaume Le Loc’h, Philipp Heeb, Mathieu Lihoreau
# The script consists of three main sections: 
# 1) R code to identify spatio-temporal behavioural patterns from complex animal trajectories 
# 2) R code to compare original tracks with Brownian walks
# 3) R code to compare original tracks with Lévy walks

### Packages needed to run the code ####

library(raster)
library(vegan)
library(dtw)
library(ggplot2)
library(Rmisc)
library(hrbrthemes)
library(igraph)
library(ggnetwork)
library(BiRewire)
library(trajr)

### function merge with order needed to run the code ####

merge.with.order <- function(x,y, ..., sort = T, keep_order)
{
  add.id.column.to.data <- function(DATA)
  {
    data.frame(DATA, id... = seq_len(nrow(DATA)))
  }
  order.by.id...and.remove.it <- function(DATA)
  {
    if(!any(colnames(DATA)=="id...")) stop("The function order.by.id...and.remove.it only works with data.frame objects which includes the 'id...' order column")
    
    ss_r <- order(DATA$id...)
    ss_c <- colnames(DATA) != "id..."
    DATA[ss_r, ss_c]
  }
  
  if(!missing(keep_order))
  {
    if(keep_order == 1) return(order.by.id...and.remove.it(merge(x=add.id.column.to.data(x),y=y,..., sort = FALSE)))
    if(keep_order == 2) return(order.by.id...and.remove.it(merge(x=x,y=add.id.column.to.data(y),..., sort = FALSE)))
    warning("The function merge.with.order only accepts NULL/1/2 values for the keep_order variable")
  } else {return(merge(x=x,y=y,..., sort = sort))}
}


#################### 1) R code to identify spatio-temporal behavioural patterns from complex animal trajectories ###############################
################## Load the dataset made of three coloums (Latitude, Longitude, ID) ####

dati<-read.table("NAME_DATASET.txt",header=T)
dati$ID<-as.factor(dati$ID)

################ Calculate step lengths between locations #######

vec<-c()
for (j in 1:length(dati[,1])){
  vec<-c(vec,sqrt((dati$Latitude[j] - dati$Latitude[j+1])^2 + 
                    (dati$Longitude[j] - dati$Longitude[j+1])^2)) 
}
dati$path<-c(0,vec[!is.na(vec)])

################ Translate spatial coordinates into temporal movement network based of step length distribution #######

step=as.numeric(quantile(dati$path,prob=c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)))
names(step)<-c("0.1_quantile","0.2_quantile","0.3_quantile","0.4_quantile","0.5_quantile",
               "0.6_quantile","0.7_quantile","0.8_quantile","0.9_quantile")
step=step[step>0]
lista_coord<-list()
lista_net<-list()
lista_point_flowers<-list()
lista_point_nest<-list()
lista_coord_nodes<-list()
lista_sequence<-list()
list_lista_net<-c()
list_lista_sequence<-c()
list_lista_coord<-c()

for(j in 1:length(step)){
  step1=step[[j]]
  for (i in 1:length(levels(dati$ID))){
    prova<-subset(dati,ID==levels(dati$ID)[i])
    prova<-data.frame(Latitude=prova$Latitude,Longitude=prova$Longitude,ID=prova$ID,path=prova$path)
    r <- raster(xmn=min(prova$Longitude)-median(prova$path), ymn=min(prova$Latitude)-median(prova$path), 
                xmx=max(prova$Longitude)+median(prova$path), ymx=max(prova$Latitude)+median(prova$path), res=step1)
    r[] <- 0
    cells<-cellFromRowColCombine(r, 1:dim(r)[1],1:dim(r)[2])
    coord_grid<-data.frame(xyFromCell(r,cells),id=cells)
    kk<-cbind(prova$Longitude,prova$Latitude)
    tomerge<-data.frame(id=unique(cellFromXY(r, kk)),new_id=rep(1:length(unique(cellFromXY(r, kk)))))
    final<-data.frame(id=cellFromXY(r, kk))
    ff=merge.with.order(final,tomerge,by="id",all.x=T, sort=F,keep_order=T)
    lista_sequence[[i]]<-ff$new_id
    from=ff$new_id[-length(ff$new_id)]
    to=ff$new_id[-1]
    net_prova<-data.frame(from,to)
    lista_coord_nodes[[i]]<-merge.with.order(ff,coord_grid,by="id",all.x=T, sort=F,keep_order=T)[,-1]
    lista_net[[i]]<-net_prova
    lista_coord[[i]]<-kk
  }
  list_lista_net[[j]]<-lista_net
  list_lista_sequence[[j]]<-lista_sequence
  list_lista_coord[[j]]<-lista_coord_nodes
}

################ prepare node sequences for motif calculation as shown in Figure 3 #######

list_lista_motivi<-list()
for (p in 1:length(step)){
  a<-list_lista_sequence[[p]][[1]]
  #a<-rle(a1)$values # remove direct loops
  lista<-list()
  lista_motivi<-list()
  j<-1
  while (j <length(dati$ID)/2){
    for (i in 1:length(list_lista_sequence[[p]][[1]])){
      if (length(unique(na.omit(a[1:i])))==3) (lista[[i]]<-a[1:i]) else (lista[[i]]<-NA)
      pp<-lista[which.max(sapply(lista,length))][[1]]
    }
    lista_motivi[[j]]<-pp[!is.na(pp)]
    j <- j + 1
    a<-a[-c(1:(length(pp)-1))]
  }
  list_lista_motivi[[p]]<-lista_motivi
}

for (p in 1:length(step)){
  list_lista_motivi[[p]]<-list_lista_motivi[[p]][lapply(list_lista_motivi[[p]],length)>0]
}

################ build networks and extract motif time-series #######

list_lista_network<-list()
list_lista_census<-list()
for(p in 1:length(step)){
  lista_network<-list()
  lista_census<-list()
  lista_motivi<-list_lista_motivi[[p]]
  for(i in 1:length(lista_motivi)){
    from=lista_motivi[[i]][-length(lista_motivi[[i]])]
    to=lista_motivi[[i]][-1]
    net<-graph_from_edgelist(cbind(from,to), directed = TRUE)
    lista_network[[i]]<-net
    lista_census[[i]]<-data.frame(count=triad_census(net)[c(6,7,8,10,11,14,15,16)],motifs_picture=rep(c(6,7,8,11,9,13,15,16)))
  }
  list_lista_network[[p]]<-lista_network
  list_lista_census[[p]]<-lista_census
}

lista_temporal_motifs<-list()
lista_only_motifs<-list()
for(p in 1:length(step)){
  lista_temporal_motifs[[p]]<-data.frame(do.call("rbind", list_lista_census[[p]]),time=rep(c(1:length(list_lista_census[[p]])),each=8))
  lista_only_motifs[[p]]<-subset(lista_temporal_motifs[[p]],count>0)
}

lista_tsData<-list()
for(p in 1:length(step)){
  lista_tsData[[p]]<-ts(lista_only_motifs[[p]]$motifs_picture)
}

################ create distance matrix between motif time-series based on DTW algorithm #######

DWT.DIST<-function (x,y)
{
  a<-na.omit(x)
  b<-na.omit(y)
  return(dtw(a,b)$normalizedDistance)
}

lista_dati_clus<-list()
for(p in 1:length(step)){
  lista_dati_clus[[p]]<-lista_only_motifs[[p]]$motifs_picture
}
distance<-dist(lista_dati_clus,method="DTW")

################ FIGURE 4: shannon diversity index to select time-series #######

data_diver<-data.frame(sequence=names(vegan::diversity(distance, index="shannon")),
                       index=as.numeric(vegan::diversity(distance, index="shannon")))
plot(data_diver$sequence,data_diver$index,
     xlab = "Motif time series", ylab = "Shannon Index")


### choose the sequence having the highest shannon diversity index ####

TS_Data<-lista_tsData[[as.numeric(which.max(data_diver$index))]]
cx=as.numeric(which.max(data_diver$index))

################ FIGURE 5: proportion of motifs #######

tt<-as.data.frame(table(lista_only_motifs[[cx]]$motifs_picture)/length(lista_only_motifs[[cx]]$motifs_picture))

#### NB! Here we have manually renamed the observed motifs. This procedure may be different depending on the dataset analyzed; ...
# ... the full possible list of motifs is the following || list(M3 = "6", M4 = "8", M5 = "7", M6="9", M8 = "11", M10 = "13", M12 = "15", M13 = "16") || ...
# ... please chack each dataset (i.e. using the command levels(tt$Var1)) and rename the following lines accordingly
# ... note that motif 7 has been renamed as M5 while motif 8 has been renamed M4. The reason is beacause motif 8 is different in complexity than motif 7 (see Figure S4 in supplementary material) 

levels(tt$Var1) <-  list(M3 = "6", M4 = "8", M5 = "7", M6="9", M8 = "11", M10 = "13")
tt$Var1 <- factor(tt$Var1, levels = c("M3","M4","M5","M6","M8","M10"))

ggplot(tt,aes(Var1,Freq))+
  geom_col(width = 0.7,fill="orange")+
  coord_flip()+
  theme(panel.background = element_blank(),
        panel.grid.major.x=element_line(color = "grey80",linetype = "dashed"),
        axis.text.x = element_text(size = 14),
        axis.title.x = element_text(size = 16),
        axis.text.y = element_text(size = 14),
        panel.border = element_rect(colour = "grey30", fill=NA, size=0.5))+
  ylab("Fraction Motifs")+
  xlab("")+
  ylim(0,1)

################ FIGURE 6: motif time series  #######

hh<-as.data.frame(TS_Data)
hh$time<-as.numeric(as.character(rownames(hh)))
hh$x<-as.factor(hh$x)
levels(hh$x) <- list(M3 = "6", M4 = "8", M5 = "7", M6 = "9", M8 = "11", M10 = "13")
hh$type=hh$x

levels(hh$type) <- list(NO = "M3", YES = "M4", YES = "M5", YES = "M6", YES = "M8", YES = "M10")
hh$time<-rep(1:length(hh[,1]))

ggplot(hh, aes(x=time, y=x, group=1)) +
  stat_summary(fun.y=sum, geom="line")+
  geom_point(stat='summary', fun.y=sum,shape=21, color="black", fill="#69b3a2", size=2) +
  geom_point(data=subset(hh,type=="YES"),aes(x=time, y=x, group=1),
             stat='summary', fun.y=sum, shape=21, color="black",fill="dodgerblue3" ,size=3) +
  theme_ipsum() +
  ylab("motifs")+
  ggtitle("Evolution of network motifs")   

################ FIGURE 6: original track coloured by motifs type #######

mot=as.numeric(lista_tsData[[cx]])

vettore<-c()
for (i in 1:length(list_lista_motivi[[cx]])){
  vettore<-c(vettore,rep(mot[i],(length(list_lista_motivi[[cx]][[i]])-1)))
}

dati_net=data.frame(y=prova$Latitude[- length(prova$Latitude)],x=prova$Longitude[- length(prova$Latitude)],
                    yend=prova$Latitude[-1],xend=prova$Longitude[-1])

dati_net$time<-rep(1:length(dati_net[,1]))
vettore=vettore[-length(vettore)]

vettore<-as.factor(as.character(vettore))
levels(vettore) <- list(NO = "6", YES = "7", YES = "8", YES = "9", YES = "11", YES = "13")
dati_net$vettore=vettore

ggplot() +
  geom_edges(data=dati_net, aes(x = x, y = y, xend = xend, yend = yend),arrow = arrow(length = unit(10, "pt"), type = "closed"),color="gray") +
  geom_nodes(data=dati_net, aes(x = x, y = y, xend = xend, yend = yend),color = "tomato", alpha=0.5, size = 3) +
  geom_nodes(data=subset(dati_net,vettore=="YES"), aes(x = x, y = y, xend = xend, yend = yend,
                                                       color = time),size = 3)+
  theme_blank()



################ FIGURE 6: original track coloured by loops #######
dato<-c()
for (i in 1:length(list_lista_motivi[[cx]])){
  dato<-rbind(dato,as.data.frame((table(list_lista_motivi[[cx]][[i]]))))
}

dato$Var1<-as.factor(as.character(dato$Var1))
dato$bin=dato$Freq
dato$bin=ifelse(dato$bin > 2, "yes", "no")   
dato = dato[!duplicated(dato$Var1),]

tomerge=data.frame(Var1=list_lista_coord[[cx]][[1]]$new_id[-1])
data_loop=merge.with.order(tomerge,dato,by="Var1",all.x=T,sort=FALSE,keep_order = 1)
dati_net$loop=data_loop$bin

ggplot() +
  geom_edges(data=dati_net, aes(x = x, y = y, xend = xend, yend = yend),arrow = arrow(length = unit(10, "pt"), type = "closed"),color="gray") +
  geom_nodes(data=dati_net, aes(x = x, y = y, xend = xend, yend = yend),color = "tomato", alpha=0.5, size = 3) +
  geom_nodes(data=subset(dati_net,loop=="yes"), aes(x = x, y = y, xend = xend, yend = yend,
                                                    color = time),size = 3)+
  theme_blank()




################ Evaluation of temporal motifs with a null model #######
# creating the probability matrix of temporal associations between motifs (i.e. p-real)#

x <- lista_only_motifs[[cx]]$motifs_picture
x<-as.numeric(as.factor(lista_only_motifs[[cx]]$motifs_picture))
p_real <- matrix(nrow = length(unique(lista_only_motifs[[cx]]$motifs_picture)), ncol = length(unique(lista_only_motifs[[cx]]$motifs_picture)), 0)
for (t in 1:(length(x) - 1)) p_real[x[t], x[t + 1]] <- p_real[x[t], x[t + 1]] + 1
for (i in 1:length(unique(lista_only_motifs[[cx]]$motifs_picture))) p_real[i, ] <- p_real[i, ] / sum(p_real[i, ])
colnames(p_real) <- sort(unique(lista_only_motifs[[cx]]$motifs_picture))
rownames(p_real) <- sort(unique(lista_only_motifs[[cx]]$motifs_picture))

# creating a list of 100 probability matrices obtained from time-shuffled time series#

lista_prob<-list()
for(j in 1:100){
  xx<-sample(as.numeric(as.factor(lista_only_motifs[[cx]]$motifs_picture)))
  p <- matrix(nrow = length(unique(lista_only_motifs[[cx]]$motifs_picture)), ncol = length(unique(lista_only_motifs[[cx]]$motifs_picture)), 0)
  for (t in 1:(length(xx) - 1)) p[xx[t], xx[t + 1]] <- p[xx[t], xx[t + 1]] + 1
  for (i in 1:length(unique(lista_only_motifs[[cx]]$motifs_picture))) p[i, ] <- p[i, ] / sum(p[i, ])
  lista_prob[[j]]<-p
}

# calculate 95%CI for each simulated pair of temporal patterns #   

n=length(lista_prob[[1]])
sim=100
output <- matrix(ncol=sim, nrow=n)
for(i in 1:sim){
  vec<-c()
  for (r in 1:nrow(lista_prob[[i]])){   
    for (c in 1:ncol(lista_prob[[i]])){  
      vec<-c(vec,lista_prob[[i]][r,c])  
    }
  }
  output[,i]<-vec}

vec_real<-c()
for (r in 1:nrow(p_real)){   
  for (c in 1:ncol(p_real)){  
    vec_real<-c(vec_real,p_real[r,c])  
  }
}

### construct a vector assigning 1 to all the probability different from random #
verify<-c()
for(i in 1:n){
  if((vec_real[i] <= CI(output[i,][!is.nan(output[i,])],ci=0.95)[[1]]) & (vec_real[i] >= CI(output[i,][!is.nan(output[i,])],ci=0.95)[[3]])) {
    verify<-c(verify,0)
  } else {
    verify<-c(verify,1)
  }
}


### construct a vector characterizing if real probabilities are occurring more than randomly expected #
sign<-c()
for(i in 1:n){
  if(vec_real[i] < CI(output[i,][!is.nan(output[i,])],ci=0.95)[[1]]) {
    sign<-c(sign,"negative")
  } else {
    sign<-c(sign,"positive")
  }
}

# construct a complete dataset for subsequent analysis 
data_randomization<-data.frame(diff_from_random=verify,motif_sequence=paste(rep(levels(as.factor(lista_only_motifs[[cx]]$motifs_picture)),each=length(tt$Var1)),
                                                                            rep(levels(as.factor(lista_only_motifs[[cx]]$motifs_picture))),sep="_"),
                               sign_comparison=sign)

from<-c()
to<-c()
for ( i in 1:length(lista_prob[[1]])){
  from<-c(from,strsplit(as.character(data_randomization$motif_sequence),"_",fixed=TRUE)[[i]][1])
  to<-c(to,strsplit(as.character(data_randomization$motif_sequence),"_",fixed=TRUE)[[i]][2])
}
data_randomization$from=from
data_randomization$to=to
data_randomization$from=as.numeric(data_randomization$from) 
data_randomization$to=as.numeric(data_randomization$to) 

## subset of probabilities 
df=subset(data_randomization,sign_comparison=="positive" & diff_from_random==1)
pp=as.matrix(table(df[,c(4,5)]))

#####
a<-c("6","8","7", "9", "11", "13", "15", "16")
#####
s=matrix(0,nrow=8,ncol=8)
colnames(s)<-a
rownames(s)<-a

cAB <- union(colnames(pp), colnames(s))
rAB <- union(rownames(pp), rownames(s))

A1 <- matrix(0, ncol=length(cAB), nrow=length(rAB), dimnames=list(rAB, cAB))
B1 <- A1

indxA <- outer(rAB, cAB, FUN=paste) %in% outer(rownames(pp), colnames(pp), FUN=paste) 
indxB <- outer(rAB, cAB, FUN=paste) %in% outer(rownames(s), colnames(s), FUN=paste)
A1[indxA] <- pp
B1[indxB] <- s

pp1=A1+B1

#################### 2) R code to compare original tracks with Brownian walks ###############################
# rename the positive non random association matrix for each dataset (e.g. if we have built the black kite and the wolf dataset we may rename each pp1 as black_kite and wolf and compare them using the following code)
black_kite=pp1 # obtained after running our method on black kite data
wolf=pp1 # obtained after running our method on wolf data


lista_black_kite<-list()
lista_wolf<-list()

for (kik in 1:100){
  
  random_walk <- function(n.org, steps, left.p = .5, up.p = .5, plot = TRUE){
    
    require(ggplot2)
    
    whereto <- matrix(ncol = 2)
    
    for(x in 1:n.org){
      walker <- matrix(c(0,0), nrow = steps+1, ncol = 2, byrow = T)
      
      for(i in 1:steps){
        # left/right = 1/0
        horizontal <- rbinom(1, 1, left.p)
        
        # distance 2
        h.dist <- abs(rnorm(1, 0, 1))
        
        # Horizontal Movement
        if(horizontal == 0){
          walker[i+1,1] <- walker[i,1] + h.dist
        }
        if(horizontal == 1){
          walker[i+1,1] <- walker[i,1] - h.dist
        }
        
        # up/down = 1/0
        vertical <- rbinom(1, 1, up.p)
        
        #distance 2
        v.dist <- abs(rnorm(1, 0, 1))
        
        # Vertical Movement
        if(vertical == 1){
          walker[i+1,2] <- walker[i,2] + v.dist
        }
        if(vertical == 0){
          walker[i+1,2] <- walker[i,2] - v.dist
        }
      }
      
      whereto <- rbind(whereto, walker)
    }
    
    id <- rep(1:n.org, each = 201)
    colnames(whereto) <- c("x" , "y")
    whereto <- as.data.frame(whereto)
    whereto <- cbind(whereto[2:nrow(whereto),], org = factor(id))
    
    if(plot){
      require(ggplot2)
      p <- ggplot(whereto, aes(x = x, y = y, colour = org))
      p <- p + geom_path()
      print(p)
    }
    
    return(whereto)
  }
  
  rw.test <- random_walk(1, 200, .5, .5)
  
  
  dati=rw.test
  
  ################## Rename tha RW dataset ####
  colnames(dati)<-c("x","y","ID")
  dati$ID<-as.factor(dati$ID)
  
  ################ Calculate step lengths between locations #######
  
  vec<-c()
  for (j in 1:length(dati[,1])){
    vec<-c(vec,sqrt((dati$y[j] - dati$y[j+1])^2 + 
                      (dati$x[j] - dati$x[j+1])^2)) 
  }
  dati$path<-c(0,vec[!is.na(vec)])
  
  ################ Translate spatial coordinates into temporal movement network based of step length distribution #######
  
  step=as.numeric(quantile(dati$path,prob=c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),na.rm=T))
  names(step)<-c("0.1_quantile","0.2_quantile","0.3_quantile","0.4_quantile","0.5_quantile",
                 "0.6_quantile","0.7_quantile","0.8_quantile","0.9_quantile")
  step=step[step>0]
  lista_coord<-list()
  lista_net<-list()
  lista_point_flowers<-list()
  lista_point_nest<-list()
  lista_coord_nodes<-list()
  lista_sequence<-list()
  list_lista_net<-c()
  list_lista_sequence<-c()
  list_lista_coord<-c()
  
  for(j in 1:length(step)){
    step1=step[[j]]
    for (i in 1:length(levels(dati$ID))){
      prova<-dati
      prova<-data.frame(Latitude=prova$x,Longitude=prova$y,ID=prova$ID,path=prova$path)
      r <- raster(xmn=min(prova$Latitude)-median(prova$path,na.rm=T), ymn=min(prova$Longitude)-median(prova$path,na.rm=T), 
                  xmx=max(prova$Latitude)+median(prova$path,na.rm=T), ymx=max(prova$Longitude)+median(prova$path,na.rm=T), res=step1)
      r[] <- 0
      cells<-cellFromRowColCombine(r, 1:dim(r)[1],1:dim(r)[2])
      coord_grid<-data.frame(xyFromCell(r,cells),id=cells)
      kk<-cbind(prova$Latitude,prova$Longitude)
      tomerge<-data.frame(id=unique(cellFromXY(r, kk)),new_id=rep(1:length(unique(cellFromXY(r, kk)))))
      final<-data.frame(id=cellFromXY(r, kk))
      ff=merge.with.order(final,tomerge,by="id",all.x=T, sort=F,keep_order=T)
      lista_sequence[[i]]<-ff$new_id
      from=ff$new_id[-length(ff$new_id)]
      to=ff$new_id[-1]
      net_prova<-data.frame(from,to)
      lista_coord_nodes[[i]]<-merge.with.order(ff,coord_grid,by="id",all.x=T, sort=F,keep_order=T)[,-1]
      lista_net[[i]]<-net_prova
      lista_coord[[i]]<-kk
    }
    list_lista_net[[j]]<-lista_net
    list_lista_sequence[[j]]<-lista_sequence
    list_lista_coord[[j]]<-lista_coord_nodes
  }
  
  ################ prepare node sequences for motif calculation as shown in Figure 3 #######
  
  list_lista_motivi<-list()
  for (p in 1:length(step)){
    a<-list_lista_sequence[[p]][[1]]
    #a<-rle(a1)$values # remove direct loops
    lista<-list()
    lista_motivi<-list()
    j<-1
    while (j <length(dati$ID)/2){
      for (i in 1:length(list_lista_sequence[[p]][[1]])){
        if (length(unique(na.omit(a[1:i])))==3) (lista[[i]]<-a[1:i]) else (lista[[i]]<-NA)
        pp<-lista[which.max(sapply(lista,length))][[1]]
      }
      lista_motivi[[j]]<-pp[!is.na(pp)]
      j <- j + 1
      a<-a[-c(1:(length(pp)-1))]
    }
    list_lista_motivi[[p]]<-lista_motivi
  }
  
  for (p in 1:length(step)){
    list_lista_motivi[[p]]<-list_lista_motivi[[p]][lapply(list_lista_motivi[[p]],length)>0]
  }
  
  ################ build networks and extract motif time-series #######
  
  list_lista_network<-list()
  list_lista_census<-list()
  for(p in 1:length(step)){
    lista_network<-list()
    lista_census<-list()
    lista_motivi<-list_lista_motivi[[p]]
    for(i in 1:length(lista_motivi)){
      from=lista_motivi[[i]][-length(lista_motivi[[i]])]
      to=lista_motivi[[i]][-1]
      net<-graph_from_edgelist(cbind(from,to), directed = TRUE)
      lista_network[[i]]<-net
      lista_census[[i]]<-data.frame(count=triad_census(net)[c(6,7,8,10,11,14,15,16)],motifs_picture=rep(c(6,7,8,11,9,13,15,16)))
    }
    list_lista_network[[p]]<-lista_network
    list_lista_census[[p]]<-lista_census
  }
  
  lista_temporal_motifs<-list()
  lista_only_motifs<-list()
  for(p in 1:length(step)){
    lista_temporal_motifs[[p]]<-data.frame(do.call("rbind", list_lista_census[[p]]),time=rep(c(1:length(list_lista_census[[p]])),each=8))
    lista_only_motifs[[p]]<-subset(lista_temporal_motifs[[p]],count>0)
  }
  
  lista_tsData<-list()
  for(p in 1:length(step)){
    lista_tsData[[p]]<-ts(lista_only_motifs[[p]]$motifs_picture)
  }
  
  ################ create distance matrix between motif time-series based on DTW algorithm #######
  
  DWT.DIST<-function (x,y)
  {
    a<-na.omit(x)
    b<-na.omit(y)
    return(dtw(a,b)$normalizedDistance)
  }
  
  lista_dati_clus<-list()
  for(p in 1:length(step)){
    lista_dati_clus[[p]]<-lista_only_motifs[[p]]$motifs_picture
  }
  distance<-dist(lista_dati_clus,method="DTW")
  
  ################ FIGURE 5: shannon diversity index to select time-series #######
  
  data_diver<-data.frame(sequence=names(vegan::diversity(distance, index="shannon")),
                         index=as.numeric(vegan::diversity(distance, index="shannon")))
  plot(data_diver$sequence,data_diver$index,
       xlab = "Motif time series", ylab = "Shannon Index")
  
  
  ### choose the sequence having the highest shannon diversity index ####
  
  TS_Data<-lista_tsData[[as.numeric(which.max(data_diver$index))]]
  cx=as.numeric(which.max(data_diver$index))
  
  ################ FIGURE 5: proportion of motifs #######
  
  tt<-as.data.frame(table(lista_only_motifs[[cx]]$motifs_picture)/length(lista_only_motifs[[cx]]$motifs_picture))
  
  #### NB! Here we have manually renamed the observed motifs. This procedure may be different depending on the dataset analyzed; ...
  # ... the full possible list of motifs is the following || list(M3 = "6", M4 = "8", M5 = "7", M6="9", M8 = "11", M10 = "13", M12 = "15", M13 = "16") || ...
  # ... please chack each dataset (i.e. using the command levels(tt$Var1)) and rename the following lines accordingly
  # ... note that motif 7 has been renamed as M5 while motif 8 has been renamed M4. The reason is beacause motif 8 is different in complexity than motif 7 (see Figure S4 in supplementary material) 
  
  levels(tt$Var1) <-  list(M3 = "6", M4 = "8", M5 = "7", M6="9", M8 = "11", M10 = "13", M12 = "15", M13 = "16")
  tt$Var1 <- factor(tt$Var1, levels = c("M3","M4","M5","M6","M8","M10","M12","M13")) 
  
  
  ################ Evaluation of temporal motifs with a null model #######
  # creating the probability matrix of temporal associations between motifs (i.e. p-real)#
  
  x <- lista_only_motifs[[cx]]$motifs_picture
  x<-as.numeric(as.factor(lista_only_motifs[[cx]]$motifs_picture))
  p_real <- matrix(nrow = length(unique(lista_only_motifs[[cx]]$motifs_picture)), ncol = length(unique(lista_only_motifs[[cx]]$motifs_picture)), 0)
  for (t in 1:(length(x) - 1)) p_real[x[t], x[t + 1]] <- p_real[x[t], x[t + 1]] + 1
  for (i in 1:length(unique(lista_only_motifs[[cx]]$motifs_picture))) p_real[i, ] <- p_real[i, ] / sum(p_real[i, ])
  colnames(p_real) <- sort(unique(lista_only_motifs[[cx]]$motifs_picture))
  rownames(p_real) <- sort(unique(lista_only_motifs[[cx]]$motifs_picture))
  p_real[is.nan(p_real)] = 0

  # creating a list of 100 probability matrices obtained from time-shuffled time series#
  
  lista_prob<-list()
  for(j in 1:100){
    xx<-sample(as.numeric(as.factor(lista_only_motifs[[cx]]$motifs_picture)))
    p <- matrix(nrow = length(unique(lista_only_motifs[[cx]]$motifs_picture)), ncol = length(unique(lista_only_motifs[[cx]]$motifs_picture)), 0)
    for (t in 1:(length(xx) - 1)) p[xx[t], xx[t + 1]] <- p[xx[t], xx[t + 1]] + 1
    for (i in 1:length(unique(lista_only_motifs[[cx]]$motifs_picture))) p[i, ] <- p[i, ] / sum(p[i, ])
    lista_prob[[j]]<-p
  }
  
  # calculate 95%CI for each simulated pair of temporal patterns #   
  
  n=length(lista_prob[[1]])
  sim=100
  output <- matrix(ncol=sim, nrow=n)
  for(i in 1:sim){
    vec<-c()
    for (r in 1:nrow(lista_prob[[i]])){   
      for (c in 1:ncol(lista_prob[[i]])){  
        vec<-c(vec,lista_prob[[i]][r,c])  
      }
    }
    output[,i]<-vec}
  
  vec_real<-c()
  for (r in 1:nrow(p_real)){   
    for (c in 1:ncol(p_real)){  
      vec_real<-c(vec_real,p_real[r,c])  
    }
  }
  
  ### construct a vector assigning 1 to all the probability different from random #
  verify<-c()
  for(i in 1:n){
    if((vec_real[i] <= CI(output[i,][!is.nan(output[i,])],ci=0.95)[[1]]) & (vec_real[i] >= CI(output[i,][!is.nan(output[i,])],ci=0.95)[[3]])) {
      verify<-c(verify,0)
    } else {
      verify<-c(verify,1)
    }
  }
  
  
  ### construct a vector characterizing if real probabilities are occurring more than randomly expected #
  sign<-c()
  for(i in 1:n){
    if(vec_real[i] < CI(output[i,][!is.nan(output[i,])],ci=0.95)[[1]]) {
      sign<-c(sign,"negative")
    } else {
      sign<-c(sign,"positive")
    }
  }
  
  # construct a complete dataset for subsequent analysis 
  data_randomization<-data.frame(diff_from_random=verify,motif_sequence=paste(rep(levels(as.factor(lista_only_motifs[[cx]]$motifs_picture)),each=length(tt$Var1)),
                                                                              rep(levels(as.factor(lista_only_motifs[[cx]]$motifs_picture))),sep="_"),
                                 sign_comparison=sign)
  
  from<-c()
  to<-c()
  for ( i in 1:length(lista_prob[[1]])){
    from<-c(from,strsplit(as.character(data_randomization$motif_sequence),"_",fixed=TRUE)[[i]][1])
    to<-c(to,strsplit(as.character(data_randomization$motif_sequence),"_",fixed=TRUE)[[i]][2])
  }
  data_randomization$from=from
  data_randomization$to=to
  data_randomization$from=as.numeric(data_randomization$from) 
  data_randomization$to=as.numeric(data_randomization$to) 
  
  ############# 
  df=subset(data_randomization,sign_comparison=="positive")
  pp=as.matrix(table(df[,c(4,5)]))
  
  #####
  a<-c("6","8","7", "9", "11", "13", "15", "16")
  #####
  s=matrix(0,nrow=8,ncol=8)
  colnames(s)<-a
  rownames(s)<-a
  
  cAB <- union(colnames(pp), colnames(s))
  rAB <- union(rownames(pp), rownames(s))
  
  A1 <- matrix(0, ncol=length(cAB), nrow=length(rAB), dimnames=list(rAB, cAB))
  B1 <- A1
  
  indxA <- outer(rAB, cAB, FUN=paste) %in% outer(rownames(pp), colnames(pp), FUN=paste) 
  indxB <- outer(rAB, cAB, FUN=paste) %in% outer(rownames(s), colnames(s), FUN=paste)
  A1[indxA] <- pp
  B1[indxB] <- s
  
  pp1=A1+B1
  pp1_matrix=as.matrix(pp1)
  
  lista_black_kite[[kik]]<-birewire.similarity(pp1_matrix,black_kite)
  lista_wolf[[kik]]<-birewire.similarity(pp1_matrix,wolf)
}

#################### 3) R code to compare original tracks with Lévy walks ###############################
############################## Building and comparing random levy models #################
# here the same code applied this time to a levy walk model #
black_kite=pp1 # obtained after running our method on black kite data
wolf=pp1 # obtained after running our method on wolf data


lista_black_kite<-list()
lista_wolf<-list()

for (kik in 1:100){


alpha=2
n=200
x=rep(0,n)
y=rep(0,n)

for (i in 2:n){
  theta=runif(1)*2*pi
  f=runif(1)^(-1/alpha)
  x[i]=x[i-1]+f*cos(theta)
  y[i]=y[i-1]+f*sin(theta)
}
coords<-data.frame(x=x,y=y)
trj <- TrajFromCoords(coords)

# Plot it
plot(trj)

################## Rename tha RW dataset ####
dati<-data.frame(coords,ID=rep(1,length(coords$x)))
colnames(dati)<-c("x","y","ID")
head(dati)
dati$ID<-as.factor(dati$ID)

################ Calculate step lengths between locations #######

vec<-c()
for (j in 1:length(dati[,1])){
  vec<-c(vec,sqrt((dati$y[j] - dati$y[j+1])^2 + 
                    (dati$x[j] - dati$x[j+1])^2)) 
}
dati$path<-c(0,vec[!is.na(vec)])

################ Translate spatial coordinates into temporal movement network based of step length distribution #######

step=as.numeric(quantile(dati$path,prob=c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),na.rm=T))
names(step)<-c("0.1_quantile","0.2_quantile","0.3_quantile","0.4_quantile","0.5_quantile",
               "0.6_quantile","0.7_quantile","0.8_quantile","0.9_quantile")
step=step[step>0]
lista_coord<-list()
lista_net<-list()
lista_point_flowers<-list()
lista_point_nest<-list()
lista_coord_nodes<-list()
lista_sequence<-list()
list_lista_net<-c()
list_lista_sequence<-c()
list_lista_coord<-c()

for(j in 1:length(step)){
  step1=step[[j]]
  for (i in 1:length(levels(dati$ID))){
    prova<-dati
    prova<-data.frame(Latitude=prova$x,Longitude=prova$y,ID=prova$ID,path=prova$path)
    r <- raster(xmn=min(prova$Latitude)-median(prova$path,na.rm=T), ymn=min(prova$Longitude)-median(prova$path,na.rm=T), 
                xmx=max(prova$Latitude)+median(prova$path,na.rm=T), ymx=max(prova$Longitude)+median(prova$path,na.rm=T), res=step1)
    r[] <- 0
    cells<-cellFromRowColCombine(r, 1:dim(r)[1],1:dim(r)[2])
    coord_grid<-data.frame(xyFromCell(r,cells),id=cells)
    kk<-cbind(prova$Latitude,prova$Longitude)
    tomerge<-data.frame(id=unique(cellFromXY(r, kk)),new_id=rep(1:length(unique(cellFromXY(r, kk)))))
    final<-data.frame(id=cellFromXY(r, kk))
    ff=merge.with.order(final,tomerge,by="id",all.x=T, sort=F,keep_order=T)
    lista_sequence[[i]]<-ff$new_id
    from=ff$new_id[-length(ff$new_id)]
    to=ff$new_id[-1]
    net_prova<-data.frame(from,to)
    lista_coord_nodes[[i]]<-merge.with.order(ff,coord_grid,by="id",all.x=T, sort=F,keep_order=T)[,-1]
    lista_net[[i]]<-net_prova
    lista_coord[[i]]<-kk
  }
  list_lista_net[[j]]<-lista_net
  list_lista_sequence[[j]]<-lista_sequence
  list_lista_coord[[j]]<-lista_coord_nodes
}

################ prepare node sequences for motif calculation as shown in Figure 3 #######

list_lista_motivi<-list()
for (p in 1:length(step)){
  a<-list_lista_sequence[[p]][[1]]
  #a<-rle(a1)$values # remove direct loops
  lista<-list()
  lista_motivi<-list()
  j<-1
  while (j <length(dati$ID)/2){
    for (i in 1:length(list_lista_sequence[[p]][[1]])){
      if (length(unique(na.omit(a[1:i])))==3) (lista[[i]]<-a[1:i]) else (lista[[i]]<-NA)
      pp<-lista[which.max(sapply(lista,length))][[1]]
    }
    lista_motivi[[j]]<-pp[!is.na(pp)]
    j <- j + 1
    a<-a[-c(1:(length(pp)-1))]
  }
  list_lista_motivi[[p]]<-lista_motivi
}

for (p in 1:length(step)){
  list_lista_motivi[[p]]<-list_lista_motivi[[p]][lapply(list_lista_motivi[[p]],length)>0]
}

################ build networks and extract motif time-series #######

list_lista_network<-list()
list_lista_census<-list()
for(p in 1:length(step)){
  lista_network<-list()
  lista_census<-list()
  lista_motivi<-list_lista_motivi[[p]]
  for(i in 1:length(lista_motivi)){
    from=lista_motivi[[i]][-length(lista_motivi[[i]])]
    to=lista_motivi[[i]][-1]
    net<-graph_from_edgelist(cbind(from,to), directed = TRUE)
    lista_network[[i]]<-net
    lista_census[[i]]<-data.frame(count=triad_census(net)[c(6,7,8,10,11,14,15,16)],motifs_picture=rep(c(6,7,8,11,9,13,15,16)))
  }
  list_lista_network[[p]]<-lista_network
  list_lista_census[[p]]<-lista_census
}

lista_temporal_motifs<-list()
lista_only_motifs<-list()
for(p in 1:length(step)){
  lista_temporal_motifs[[p]]<-data.frame(do.call("rbind", list_lista_census[[p]]),time=rep(c(1:length(list_lista_census[[p]])),each=8))
  lista_only_motifs[[p]]<-subset(lista_temporal_motifs[[p]],count>0)
}

lista_tsData<-list()
for(p in 1:length(step)){
  lista_tsData[[p]]<-ts(lista_only_motifs[[p]]$motifs_picture)
}

################ create distance matrix between motif time-series based on DTW algorithm #######

DWT.DIST<-function (x,y)
{
  a<-na.omit(x)
  b<-na.omit(y)
  return(dtw(a,b)$normalizedDistance)
}

lista_dati_clus<-list()
for(p in 1:length(step)){
  lista_dati_clus[[p]]<-lista_only_motifs[[p]]$motifs_picture
}
distance<-dist(lista_dati_clus,method="DTW")

################ FIGURE 5: shannon diversity index to select time-series #######

data_diver<-data.frame(sequence=names(vegan::diversity(distance, index="shannon")),
                       index=as.numeric(vegan::diversity(distance, index="shannon")))
plot(data_diver$sequence,data_diver$index,
     xlab = "Motif time series", ylab = "Shannon Index")


### choose the sequence having the highest shannon diversity index ####

TS_Data<-lista_tsData[[as.numeric(which.max(data_diver$index))]]
cx=as.numeric(which.max(data_diver$index))

################ FIGURE 5: proportion of motifs #######

tt<-as.data.frame(table(lista_only_motifs[[cx]]$motifs_picture)/length(lista_only_motifs[[cx]]$motifs_picture))

#### NB! Here we have manually renamed the observed motifs. This procedure may be different depending on the dataset analyzed; ...
# ... the full possible list of motifs is the following || list(M3 = "6", M4 = "8", M5 = "7", M6="9", M8 = "11", M10 = "13", M12 = "15", M13 = "16") || ...
# ... please chack each dataset (i.e. using the command levels(tt$Var1)) and rename the following lines accordingly
# ... note that motif 7 has been renamed as M5 while motif 8 has been renamed M4. The reason is beacause motif 8 is different in complexity than motif 7 (see Figure S4 in supplementary material) 

levels(tt$Var1) <-  list(M3 = "6", M4 = "8", M5 = "7", M6="9", M8 = "11", M10 = "13", M12 = "15", M13 = "16")
tt$Var1 <- factor(tt$Var1, levels = c("M3","M4","M5","M6","M8","M10","M12","M13")) 


################ Evaluation of temporal motifs with a null model #######
# creating the probability matrix of temporal associations between motifs (i.e. p-real)#

x <- lista_only_motifs[[cx]]$motifs_picture
x<-as.numeric(as.factor(lista_only_motifs[[cx]]$motifs_picture))
p_real <- matrix(nrow = length(unique(lista_only_motifs[[cx]]$motifs_picture)), ncol = length(unique(lista_only_motifs[[cx]]$motifs_picture)), 0)
for (t in 1:(length(x) - 1)) p_real[x[t], x[t + 1]] <- p_real[x[t], x[t + 1]] + 1
for (i in 1:length(unique(lista_only_motifs[[cx]]$motifs_picture))) p_real[i, ] <- p_real[i, ] / sum(p_real[i, ])
colnames(p_real) <- sort(unique(lista_only_motifs[[cx]]$motifs_picture))
rownames(p_real) <- sort(unique(lista_only_motifs[[cx]]$motifs_picture))
p_real[is.nan(p_real)] = 0

# creating a list of 100 probability matrices obtained from time-shuffled time series#

lista_prob<-list()
for(j in 1:100){
  xx<-sample(as.numeric(as.factor(lista_only_motifs[[cx]]$motifs_picture)))
  p <- matrix(nrow = length(unique(lista_only_motifs[[cx]]$motifs_picture)), ncol = length(unique(lista_only_motifs[[cx]]$motifs_picture)), 0)
  for (t in 1:(length(xx) - 1)) p[xx[t], xx[t + 1]] <- p[xx[t], xx[t + 1]] + 1
  for (i in 1:length(unique(lista_only_motifs[[cx]]$motifs_picture))) p[i, ] <- p[i, ] / sum(p[i, ])
  lista_prob[[j]]<-p
}

# calculate 95%CI for each simulated pair of temporal patterns #   

n=length(lista_prob[[1]])
sim=100
output <- matrix(ncol=sim, nrow=n)
for(i in 1:sim){
  vec<-c()
  for (r in 1:nrow(lista_prob[[i]])){   
    for (c in 1:ncol(lista_prob[[i]])){  
      vec<-c(vec,lista_prob[[i]][r,c])  
    }
  }
  output[,i]<-vec}

vec_real<-c()
for (r in 1:nrow(p_real)){   
  for (c in 1:ncol(p_real)){  
    vec_real<-c(vec_real,p_real[r,c])  
  }
}

### construct a vector assigning 1 to all the probability different from random #
verify<-c()
for(i in 1:n){
  if((vec_real[i] <= CI(output[i,][!is.nan(output[i,])],ci=0.95)[[1]]) & (vec_real[i] >= CI(output[i,][!is.nan(output[i,])],ci=0.95)[[3]])) {
    verify<-c(verify,0)
  } else {
    verify<-c(verify,1)
  }
}


### construct a vector characterizing if real probabilities are occurring more than randomly expected #
sign<-c()
for(i in 1:n){
  if(vec_real[i] < CI(output[i,][!is.nan(output[i,])],ci=0.95)[[1]]) {
    sign<-c(sign,"negative")
  } else {
    sign<-c(sign,"positive")
  }
}

# construct a complete dataset for subsequent analysis 
data_randomization<-data.frame(diff_from_random=verify,motif_sequence=paste(rep(levels(as.factor(lista_only_motifs[[cx]]$motifs_picture)),each=length(tt$Var1)),
                                                                            rep(levels(as.factor(lista_only_motifs[[cx]]$motifs_picture))),sep="_"),
                               sign_comparison=sign)

from<-c()
to<-c()
for ( i in 1:length(lista_prob[[1]])){
  from<-c(from,strsplit(as.character(data_randomization$motif_sequence),"_",fixed=TRUE)[[i]][1])
  to<-c(to,strsplit(as.character(data_randomization$motif_sequence),"_",fixed=TRUE)[[i]][2])
}
data_randomization$from=from
data_randomization$to=to
data_randomization$from=as.numeric(data_randomization$from) 
data_randomization$to=as.numeric(data_randomization$to) 

############# 
df=subset(data_randomization,sign_comparison=="positive")
pp=as.matrix(table(df[,c(4,5)]))

#####
a<-c("6","8","7", "9", "11", "13", "15", "16")
#####
s=matrix(0,nrow=8,ncol=8)
colnames(s)<-a
rownames(s)<-a

cAB <- union(colnames(pp), colnames(s))
rAB <- union(rownames(pp), rownames(s))

A1 <- matrix(0, ncol=length(cAB), nrow=length(rAB), dimnames=list(rAB, cAB))
B1 <- A1

indxA <- outer(rAB, cAB, FUN=paste) %in% outer(rownames(pp), colnames(pp), FUN=paste) 
indxB <- outer(rAB, cAB, FUN=paste) %in% outer(rownames(s), colnames(s), FUN=paste)
A1[indxA] <- pp
B1[indxB] <- s

pp1=A1+B1
pp1_matrix=as.matrix(pp1)

lista_black_kite[[kik]]<-birewire.similarity(pp1_matrix,black_kite)
lista_wolf[[kik]]<-birewire.similarity(pp1_matrix,wolf)
}
