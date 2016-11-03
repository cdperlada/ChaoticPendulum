#!/usr/bin/env python

"""
Code for Chaotic Pendulum (W.Kinzel/G.Reents, Physics by Computer)

This code is based on pendulum.c listed in Appendix E of the book and will
replicate Fig. 4.3 of the book.
"""

__author__ = "Christian Alis"
__credits__ = "W.Kinzel/G.Reents"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import sin, cos
from scipy.integrate import odeint

class Simulation(object):
    r = 0.25
    a = 0.7
    lims = 3.0
    dt = 0.1
    pstep = 3*np.pi
    poncaire = False

    def derivs(self, t, y):
        # add a docstring to this function
        '''
        Return first time derivative of phi and omega.
        
        Parameters
        ----------
        t : float
            Time
        y : array_like
            Contains phi and omega            
        '''
        return np.array([y[1], - self.r*y[1] - sin(y[0]) + self.a*cos(2./3.*t)])
    
    def rk4_step(self, f, x, y, h):
        # add a docstring to this function
        '''
        Perform a 4th order Runge-Kutta (RK4) to solve differential equation.
        
        Parameters
        ----------
        f : function
            Differential equation to be solved.
        x : float
            Argument of function f. Function f is differentiated with respect to x.
        y : float
            Argument of function f.
        h : float
            Step size for the RK4.
        '''
        k1 = h*f(x, y)
        k2 = h*f(x + 0.5*h, y + 0.5*k1)
        # complete the following lines
        k3 = h*f(x + 0.5*h, y + 0.5*k2)
        k4 = h*f(x + h, y + 0.5*k3)
        return np.array(y + k1/6. + k2/3. + k3/3. + k4/6.)
    
    def update(self, frame, line, f, x, h, pstep):
        # add a docstring to this function
        '''
        Solve phi and omega from the differential equations and plot the corresponding points.
        
        Parameters
        ----------
        line : matplotlib.lines.Line2D      
            Contains data for plotting.
        f : function
            Differential equation to be solved.
        x : float
            Argument of function f. Function f is differentiated with respect to x.
        h : float
            Step size for the RK4.
        pstep : float
            Step size for the RK4.
        '''
        if self.poncaire:
            self.y.append(self.rk4_step(f, frame*h, self.y[-1], pstep))
        else:
            self.y.append(self.rk4_step(f, frame*h, self.y[-1], h))
        xs, ys = zip(*self.y)
        line.set_xdata(xs)
        line.set_ydata(ys)
    
    def continue_loop(self):
        # add a docstring to this function
        '''
        Create a function generator.
        '''
        i = 0
        while 1:
            i += 1
            # what does yield do?
            '''It outputs a generator instead of an iterable'''
            
            # what will happen if return is used instead of yield?
            '''The animation will not run since the program will immediately get out 
            of the function once i is returned.'''
            
            yield i
    
    def on_key(self, event):
        key = event.key
        if key == 'i':
            self.a += 0.01
            self.info_text.set_text("$r$ = %0.2f\t$a$ = %0.6f" % (self.r,
                                                                  self.a))
        # add analogous code such that pressing "d" will decrease a by 0.01
        elif key == 'd':
            self.a -= 0.01
            self.info_text.set_text("$r$ = %0.2f\t$a$ = %0.6f" % (self.r,
                                                                  self.a))

        elif key == '+':
            self.lims /= 2
            plt.xlim(-self.lims, self.lims)
            plt.ylim(-self.lims, self.lims)
        # add analogous code such that pressing "-" will zoom out the plot
        elif key == '-':
            self.lims *= 2
            plt.xlim(-self.lims, self.lims)
            plt.ylim(-self.lims, self.lims)

        elif key == 't':
            self.poncaire = not self.poncaire
            if self.poncaire:
                self.line.set_linestyle('')
                self.line.set_marker('.')
                self.y = [np.array([np.pi/2, 0])]
            else:
                self.line.set_linestyle('-')
                self.line.set_marker('')
                self.y = [np.array([np.pi/2, 0])]
    
    def run(self):
        fig = plt.figure()
        # what are the other possible events?
        '''button_press_event, button_release_event, draw_event, key_release_event, etc.'''
        
        fig.canvas.mpl_connect('key_press_event', self.on_key)
    
        self.y = [np.array([np.pi/2, 0])]
        self.line, = plt.plot([], [])
        plt.xlim(-self.lims, self.lims)
        plt.ylim(-self.lims, self.lims)
        self.info_text = plt.figtext(0.15, 0.85, "$r$ = %0.2f\t$a$ = %0.6f" % (self.r, self.a))
        anim = FuncAnimation(fig, self.update, self.continue_loop, 
                             fargs=(self.line, self.derivs, 0, self.dt, self.pstep),
                             interval=25, repeat=False)
        plt.show()

if __name__ == "__main__":
    sim = Simulation()
    sim.run()
