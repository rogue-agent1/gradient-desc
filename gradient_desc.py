#!/usr/bin/env python3
"""gradient_desc - Gradient descent optimizer."""
import sys,argparse,json,math
def rosenbrock(x,y):return(1-x)**2+100*(y-x**2)**2
def rosenbrock_grad(x,y):return[-2*(1-x)-400*x*(y-x**2),200*(y-x**2)]
def himmelblau(x,y):return(x**2+y-11)**2+(x+y**2-7)**2
def optimize(fn,grad_fn,start,lr=0.001,steps=1000):
    x,y=start;history=[]
    for i in range(steps):
        val=fn(x,y);gx,gy=grad_fn(x,y)
        history.append({"step":i,"x":round(x,6),"y":round(y,6),"value":round(val,6)})
        x-=lr*gx;y-=lr*gy
        if abs(gx)<1e-8 and abs(gy)<1e-8:break
    return x,y,fn(x,y),history
def main():
    p=argparse.ArgumentParser(description="Gradient descent")
    p.add_argument("--fn",choices=["rosenbrock"],default="rosenbrock")
    p.add_argument("--lr",type=float,default=0.001);p.add_argument("--steps",type=int,default=5000)
    p.add_argument("--start",nargs=2,type=float,default=[-1.0,1.0])
    args=p.parse_args()
    x,y,val,history=optimize(rosenbrock,rosenbrock_grad,args.start,args.lr,args.steps)
    print(json.dumps({"function":args.fn,"minimum":{"x":round(x,6),"y":round(y,6),"value":round(val,6)},"steps":len(history),"lr":args.lr,"sample":history[::max(1,len(history)//10)]},indent=2))
if __name__=="__main__":main()
