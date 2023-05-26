import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
def bayesian_p(nhyp:int, elem:int, b:list, h:list):
    """Calculates bayesian inference"""
    p_hyp_d =[]
    for i in range(nhyp):
        p_hyp_d.append(h[i]*b[i][elem])
    a = 1/(sum(p_hyp_d))
    p_hyp_d_norm = [round(i * a, 5) for i in p_hyp_d]
    return p_hyp_d_norm

def animate():
    def update(frame):
        """Update function for animation"""
        global count
        global drawpoint
        global selected
        names=name+[]
        #Updates various values inside plot
        for i in range(len(plots)):
            plots[i].set_xdata(x[:frame])
            plots[i].set_ydata(story[i][:frame])
            try:
                #Updates labels to include probability over draws
                names[i]=name[i]+" --- "+str(story[i][frame])
                try:
                    #Keeps track of box with highest probability.
                    #If a box has >=0.95 probability for 20 draws then it is chosen
                    if story[i][frame]>=0.95 and drawpoint==0:
                        count[i]+=1
                        if count[i]==20:
                            drawpoint=frame
                            selected=i
                except: pass
            except: pass
        ax.legend(labels=names)
        return plots
    #Random numbers decide how many box and candies there are
    nboxes=6
    ncandies=3

    #H = Probability of each box
    h=[random.randint(0,100) for i in range(nboxes)]
    h=[round(h[i]/sum(h),5) for i in range(len(h))]

    #B = Probability inside each box
    b=[]
    for i in range(nboxes):
        for j in range(ncandies):
            k=[random.randint(0,100) for i in range(ncandies)]
            b.append([round(k[i]/sum(k),5) for i in range(len(k))])
    #D is the dataset. 0 is first candy, 1 is second candy...
    d = []

    #Story keeps track of probability values
    story=[]
    for i in range(len(h)):
        story.append([])
    name=["Hyp"+str(i+1) for i in range(len(h))]

    #Draws indicates max number of candy being drawn out
    draws=150
    x=np.linspace(0,draws,draws)

    #Text displays box info on figure
    text=""
    for i in range(len(name)):
        text=text+f"{name[i]} --- "
        for j in range(len(b[i])):
            text=text+f"{j+1} - {round(b[i][j]*100, 3)}%, "
        text=text[:-2]+".\n"

    #Box choses random box
    box=random.randint(0,nboxes-1)

    #Creates new data
    for i in range(draws):
        d.append((random.choices([x for x in range(ncandies)],b[box])[0]))

        #Calculates new probability and updates story
        h = bayesian_p(len(h),d[i],b, h)
        for j in range(len(h)):
            story[j].append(h[j])

    #Shows new probility distribution. Various values used in graph
    fig=plt.figure(figsize=(8,6))
    sc=1.8
    panel_width = 1/sc
    panel_height = 1/sc
    off = (1 - 1/sc) / 2
    ax=fig.add_axes([off,0.37,panel_width,panel_height])

    #Creates a list of plots object to update in animation
    plots=[]
    for i in range(len(story)):
        plots.append(ax.plot(x[0],story[i][0],label=f"{name[i]}")[0])
    ax.set(xlim=[0,draws],ylim=[0,1],xlabel="Number of draws", ylabel="Probability of box")
    ax.legend()
    plt.title(f"Probability over number of new data --- Box is number {box+1}")
    plt.subplots_adjust(wspace=1)
    ax.annotate(text,  xy=(0.5, 0), xytext=(0, 10), xycoords=('axes fraction', 'figure fraction'), textcoords='offset points', ha="center", va="bottom")

    #Animates probability graph
    count=[0]*nboxes
    selected=0
    drawpoint=0
    ani=animation.FuncAnimation(fig=fig, func=update, frames=draws+20, interval=10, repeat=True, repeat_delay=10)

    #Prints output
    if drawpoint!=0:
        print(f"Box {box+1} was chosen. After {drawpoint} draws, box {selected+1} was selected with probability of {story[selected][drawpoint]}")
    else:
        lastframe=[]
        for i in story:
            lastframe.append(i[-1])
        highest=lastframe.index(max(lastframe))
        print(f"Box {box+1} was chosen. After {draws} draws, box {highest+1} was the box with highest probability of {story[highest][-1]}")
        if box!=highest:
            print(f"For reference:\nBox {box+1} - {b[box]}\nBox {highest+1} - {b[highest]}")
    return ani