library(ggplot2)
library(BSDA)
library(sjmisc)
lgrid <- matrix(NA, nrow = 8, ncol = 8)
lgrid[1,] <- c("r", "l", "q", "s", "t", "z", "c", "a")
lgrid[2,] <- c("i", "v", "d", "z", "h", "l", "t", "p")
lgrid[3,] <- c("u", "r", "o", "y", "w", "c", "a", "c")
lgrid[4,] <- c("x", "r", "f", "n", "d", "p", "g", "v")
lgrid[5,] <- c("h", "j", "f", "f", "k", "h", "g", "m")
lgrid[6,] <- c("k", "y", "e", "x", "x", "g", "k", "i")
lgrid[7,] <- c("l", "q", "e", "q", "f", "u", "e", "b")
lgrid[8,] <- c("l", "s", "d", "h", "i", "k", "y", "n")


move_function<- function(curr_sq){
  row_num=curr_sq[1]
  col_num=curr_sq[2]
  if(any(row_num==c(1,8))|any(col_num==c(1,8))) {  # If on the edge, select random square
    row_num_new<- sample(1:8, size = 1) 
    col_num_new<- sample(1:8, size = 1)
    } 
  else{
    row_num_new<-sample(c(row_num-1,row_num,row_num+1),size=1)
    if (row_num_new==row_num) # To prevent landing on same square and move to adjacent square
      col_num_new<-sample(c(col_num-1,col_num+1),size=1)
    else{
      col_num_new<-sample(c(col_num-1,col_num,col_num+1),size=1)}
  }
  next_sq<-c(row_num_new,col_num_new)
  return(next_sq)
}

green_sq<- function(row_num,col_num){
  if (any(row_num==c(2,3,6,7)) & (abs(col_num-row_num)==4)) {
    return('yes')
  }
  else {
    return('no')
  }
}

moves_for_palindrome<- function(start_sq,p){
  row_start=unlist(strsplit(start_sq,""))[1]
  col_num_start=strtoi(unlist(strsplit(start_sq,""))[2])
  moves=1
  coll<-list()
  tot<-list()
  row_num=which(letters == tolower(row_start))
  col_num=col_num_start
  coll<-c(coll,lgrid[row_num,col_num])
  #tot<-c(coll,lgrid[row_num,col_num])
  while(length(coll)<5){
    coll<-unlist(coll)
    if (green_sq(row_num,col_num)=='yes') { # Green square
      choice=sample(c(1,2),size=1,prob=c(p,1-p))
      if (choice==1){ # If choice 1 with probability of p
        coll=c('f','f','h','k')
      }
      else{ # If choice 2 with probability of 1-p
        coll<-coll[coll!=lgrid[row_num,col_num]]
      }}
    tot<-unlist(tot)
    curr_sq<-c(row_num,col_num)
    new_sq=move_function(curr_sq)
    moves=moves+1
    row_num=new_sq[1]
    col_num=new_sq[2]
    alphabet<-lgrid[row_num,col_num]
    tot<-c(tot,alphabet)
    if (length(coll)<2){
      if(is.element(alphabet,coll)) {coll<-c(coll,alphabet)}
      else if(moves<=5 & table(lgrid)[alphabet] >= 3) {coll<-c(coll,alphabet)}
      else if(moves>5 & table(lgrid)[alphabet] >= 2) {coll<-c(coll,alphabet)}
    }
    else if(length(coll)==2){
      if(is.element(alphabet,coll)) {coll<-c(coll,alphabet)}
      else if(moves<10 & table(lgrid)[alphabet] >= 2) {coll<-c(coll,alphabet)}
      else if(moves>9) {coll<-c(coll,alphabet)}
      }
    else if(length(coll)==3) {
      if(any(table(coll)>=2)|is.element(alphabet,coll)){
      coll<-c(coll,alphabet)
    }}
    else if (length(coll)==4) {
      if (!is.element(coll[4],coll[1:3])){ 
        if (is.element(alphabet,coll) & (table(coll)[alphabet]!=c(2))) {
          coll<-c(coll,alphabet)
        }}
      else if (all(table(coll)==c(2))) {
        coll<-c(coll,alphabet)
      }
      else if (is.element(alphabet,coll) & alphabet!=coll[4]) {
        coll<-c(coll,alphabet)
      }
    }
  }
  #return(moves)
  return(list(moves,coll,tot))
}
?is.element
set.seed(210116270)
rand_row<-sample(LETTERS[2:7],size=1)
rand_col<-sample(2:7,size=1)
rand_prob<- sample(seq(0,1,by=0.01),size = 1)
rand_sq<-paste(rand_row, rand_col, sep = "")
coll<- moves_for_palindrome(rand_sq,rand_prob)[2]
moves<- moves_for_palindrome(rand_sq,rand_prob)[1]
print(moves)
cat("The number of moves required to form a palindrome is=",moves,"and the letters
      in collection are:-")
print(coll)
set.seed(210116270)
prob_list<-seq(0,1,by=0.2)
mean_moves<-rep(NA,length(prob_list))
j=0
for (prob in prob_list) {
  j=j+1
  mean_moves[j]<-mean((replicate(10000,moves_for_palindrome('D4',prob))))
  cat("For probability",prob,",: Average number of moves needed for palindrome from D4 is=", 
      mean_moves[j],"\n")
}
data<- data.frame(prob_list,mean_moves)
ggplot(data, aes(x=prob_list, y=mean_moves)) +
  geom_line(color="blue", size=1.5, alpha=0.6, linetype=2) + xlab("Probabilities")+ 
  ylab("Average number of moves") + ggtitle("Average number of moves required to form palindrome against various probability values")
ggplot(data, aes(x=prob_list, y=mean_moves)) +
  geom_line(color="blue") +
  ggtitle("Average number of moves required to form palindrome against p")

set.seed(210116270)
m<- replicate(10000,moves_for_palindrome('D4',0.95))
hga<-hist(m)
n<- replicate(10000,moves_for_palindrome('F6',0.05))
hgb<-hist(n,breaks=200)
z.test(m,sigma.x=1,n,sigma.y=1,mu=0,alternative = "two.sided")
ks.test(m,n)
t.test(m,n,alternative = "two.sided", paired = F)
qqplot(m,n)
mt<- data.frame(m,n)
ggplot() + 
  stat_qq(aes(sample = m), colour = "green") + 
  stat_qq(aes(sample = n), colour = "red")

descr(mt)
plot(hga,col=rgb(0,0,1,1/4), xlim=c(0,300))
plot(hgb,col=rgb(1,0,0,1/4), xlim=c(0,300), add=T)
mean(m)
sd(m)
mean(n)
sd(n)
quantile(m,0.25)
quantile(n,0.25)
quantile(m, probs = c(0.25, 0.75))
quantile(n, probs = c(0.25, 0.75))
sort(n,decreasing = TRUE)
sort(m,decreasing = TRUE)

sq_A <- c(25,13,16,24,11,12,24,26,15,19,34)
sq_B <- c(35,41,23,26,18,15,33,42,18,47,21,26)
t.test(x = sq_A, y = sq_B, alternative = "two.sided", paired = F)

coll<-c('f','p','d','k')
!is.element(coll[4],coll[1:3])
(table(coll)>=(2))
(table(coll)['b'] == 2)
table(coll)==c(2)
coll[4]!=coll[1:3]
coll[coll!=coll[table(coll)==c(2)]]

a<- c('D',4)
a[1]
a[2]
