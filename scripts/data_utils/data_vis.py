### This plot is for data visualization ###
import numpy as np
import matplotlib.pyplot as plt
import os
### Visualize the velocity map plot
def plot_single_velocity(data):
    plt.imshow(data[0,0,].detach().cpu().numpy(), cmap='jet')
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Depth (m)')
    plt.colorbar()
    plt.show()
                
### Visualize the seismic data
def plot_single_seismic_2(data):
    nz, nx = data.shape
    plt.rcParams.update({'font.size': 18})
    vmin, vmax = np.min(data), np.max(data)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.matshow(data, aspect='auto', cmap='gray', vmin=-1, vmax=2)
    ax.set_aspect(aspect=nx/nz)
    ax.set_xticks(range(0, nx, int(300//(1050/nx)))[:5])
    ax.set_xticklabels(range(0, 1050, 300))
    ax.set_title('Offset (m)', y=1.08)
    ax.set_yticks(range(0, nz, int(200//(1000/nz)))[:5])
    ax.set_yticklabels(range(0, 1000, 200))
    ax.set_ylabel('Time (ms)', fontsize=18)
    fig.colorbar(im, ax=ax, shrink=1.0, pad=0.01, label='Amplitude')
    plt.show()

def plot_single_v(data):
    fig, ax = plt.subplots()  
    cax = ax.imshow(data[0, 0].detach().cpu().numpy(), cmap='jet')
    
    # Set labels
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Depth (m)')
    
    # Add colorbar
    fig.colorbar(cax, ax=ax)
    
    plt.show()
    
def plot_single(data, path):
    #os.mkdir(path)
    nz, nx = data.shape
    plt.rcParams.update({'font.size': 18})
    vmin, vmax = np.min(data), np.max(data)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.matshow(data, aspect='auto', cmap='gray', vmin=-1, vmax=2)
    
    # Set the aspect ratio to maintain correct stretching
    ax.set_aspect(aspect=nx / nz)
    
    # Configure x-axis to span from 0 to 700 meters with matching ticks and labels
    num_ticks = 5  # Adjust this based on the desired number of ticks
    x_ticks = np.linspace(0, nx - 1, num_ticks).astype(int)
    x_labels = np.linspace(0, 700, num_ticks).astype(int)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.xaxis.set_ticks_position('bottom')  # Move x-axis ticks to the bottom
    ax.set_xlabel('Length (m)')
    
    # Configure y-axis (Time in milliseconds)
    ax.set_yticks(range(0, nz, int(200 // (1000 / nz)))[:5])
    ax.set_yticklabels(range(0, 1000, 200))
    ax.set_ylabel('Time (ms)', fontsize=18)
    #plt.show(fig)
    # Save the plot with zero margin
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    #plt.close(fig)
