/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * Apache License, Version 2.0 
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Use this file to test new functionality
 *
 ******************************************************************************/

#include "levelset.h"
#include "commonkernels.h"
#include "particle.h"
#include "shapes.h"
#include <cmath>
#include <fstream>

using namespace std;
#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif
#ifndef M_E
#define M_E 2.718281828459045235360287471352662497
#endif

namespace Manta {

// simple 1D advection-diffusion and Burger's equation tests

// periodic grid access (2D x=space, y=time)
static Real& get(Grid<Real>& u, int i, int j , int k) {
	const int s = u.getSizeX();
	int io=i,jo=j,ko=k;

	// periodic space (=x)
	while(i<0)  i+=s;
	while(i>=s) i-=s;

	// periodic time
	while(j<0)  j+=s;
	while(j>=s) i-=s;

	//std::cerr <<" 	acc "<<io<<","<<jo<<","<<ko<<"  ->   "<<i<<","<<j<<","<<k <<"\n";
	return u( i, j, k);
}

// unused (helper to average / smoothen point at ijk with i+1 and i-1, note - no copy, same grid!)
static void smoothenX(Grid<Real>& u, int i, int j , int k) {
	get(u,i,j,k) = 0.5 * ( get(u,i+1,j,k) + get(u,i-1,j,k) );
}

// clamp for stabilization
static void clamp(Real& u, Real v) {
	if(u>v)  u = v;
	if(u<-v) u = -v;
}

// initialize first state of simulation
static void initT0(FlagGrid& flags, Grid<Real>& u, Grid<Real> *dens=NULL,
					Real f1=0., Real f2=1., Real f3=1.,  Real f4=1.,  Real f5=1. , Real f6=1. , Real f7=1., 
					Real off1=0. , Real off2=0., Real offd=0., Real fd=1. )
{
	const int t0 = flags.getParent()->mFrame;
	const int s  = flags.getSizeX();
	int       t  = t0; while(t>=s) t-=s;
	
	FOR_IJK(flags)
	{
		if(t!=j) continue;
		if (!flags.isFluid(i,j,k)) continue; // needed? currently unused
		
		Real& v = u(i,j,k);
		float d = float(i) / float(s);
		v = f1 + f2*sin(d * 2. * M_PI + off1) + f3*sin(d * 4. * M_PI + off2)+ f4*sin(d * 8. * M_PI + off1*0.4)+ f5*sin(d * 16. * M_PI + off2*0.3);

		if(dens) {
			(*dens)(i,j,k) = sin(d * fd * 2. * M_PI * 12. + offd);
			(*dens)(i,j,k) *= (*dens)(i,j,k) * (*dens)(i,j,k); // cubed, sharper transitions
		}
	}	
}

// continuous forcing, similar to initT0
void addForcing(FlagGrid& flags, Grid<Real>& u, 
					Real f1=0., Real f2=1., Real f3=1.,  Real f4=1.,  Real f5=1. , Real f6=1. , Real f7=1., 
					Real off1=0. , Real off2=0., Real offd=0.)
{
	const int t0 = flags.getParent()->mFrame;
	const int s  = flags.getSizeX();
	int       t  = t0; while(t>=s) t-=s;

	FOR_IJK(flags)
	{
		if(t!=j) continue;
		if (!flags.isFluid(i,j,k)) continue; 
		
		Real& v = u(i,j,k);
		float timeoff = 0.5 * (t0*f7) + 1.0 * t0*f7*sin(float(t) / float(s) * 3. );
		//float timeoff = t0*f7*sin(float(t) / float(s) * 2. );
		float d = (float(i)+timeoff) / float(s);  // offset with t0 over time , f7 controls direction and speed
		v += f6 * ( f2*sin(d * 2. * M_PI) + f3*sin(d * 4. * M_PI+ off1)+ f4*sin(d * 8. * M_PI+ off2)+ f5*sin(d * 16. * M_PI+ off1*0.7) );
	}
}


//! compute 1st / 2nd derivatives of u, note velocity and u can be same grid
void computeDerivatives(int methodDeriv1st, int methodDeriv2nd, FlagGrid& flags, Grid<Real>& vel, Grid<Real>& u, Grid<Real>& g1, Grid<Real>& g2)
{
	const int t0 = flags.getParent()->mFrame;
	const int s  = flags.getSizeX();
	int       t  = t0; while(t>=s) t-=s;
	const Real dx = 1. / Real(u.getSizeX());


	//comp g1,g2  -- ugly, for "j-1"
	FOR_IJK(flags) {
		if(t!=j) continue;		

		switch(methodDeriv1st) {
			case 0: { // central difference
				g1(i,j,k) = (get(u, i+1,j-1,k) - get(u, i-1,j-1,k)) / (2.*dx);
			} break;

			case 1: { // first order upwinding
				if( get(vel, i,j-1,k)>0. ) {
					g1(i,j,k) = (get(u, i+0,j-1,k) - get(u, i-1,j-1,k)) / (1.*dx);
				} else {
					g1(i,j,k) = (get(u, i+1,j-1,k) - get(u, i-0,j-1,k)) / (1.*dx);
				}				
			} break;

			default:
				errMsg("Unknown derivative method (g1)"<<methodDeriv1st);
		}

		switch(methodDeriv2nd) { // no options for now...
			case 0: { // central difference
				g2(i,j,k) = (get(u, i+1,j-1,k) - 2.*get(u, i,j-1,k) + get(u, i-1,j-1,k)) / (dx*dx);
			} break;

			case 1: { // implicit, external - nothing to do...
				//g2(i,j,k) = (get(u, i+1,j-1,k) - 2.*get(u, i,j-1,k) + get(u, i-1,j-1,k)) / (dx*dx);
			} break;

			default:
				errMsg("Unknown derivative method (g2) "<<methodDeriv2nd);
		}
		
		// simple stabilization
		clamp( g1(i,j,k) , 2.*s );
		clamp( g2(i,j,k) , 2.*s*2.*s);
	}
}

// u velocity, g1/g2 1st 2nd derivs
// j is time (bottom to top)
// f1-5 are sine curve parameters for init & forcing , 
// f6 strength of forcing, off1-2 are offsets that are reused several times with scaling
// methodDeriv1st = method to compute 1st derivative for advection  
//                  0 central (2nd), 1 upwind (1st)
// methodDeriv2nd = method to compute 2nd derivatives for diffusion
//                  0 central (2nd), 1 implicit central (operator splitting, external, relies on solveDiffusion1D)
PYTHON() void burger(FlagGrid& flags, Grid<Real>& u, Grid<Real>& g1, Grid<Real>& g2 ,
					int methodDeriv1st=0, int methodDeriv2nd=0, Real nu=0.05,
					Real f1=0., Real f2=1., Real f3=1.,  Real f4=1.,  Real f5=1. , Real f6=1. , Real f7=1., 
					Real off1=0. , Real off2=0. , 
					bool contForcing=false)
{
	const int t0 = flags.getParent()->mFrame;
	const int s  = flags.getSizeX();
	int       t  = t0; while(t>=s) t-=s;

	const Real dx = 1. / Real(u.getSizeX());
	const Real dt = flags.getParent()->mDt;

	// init sines , burger
	if(t0==0) {
		initT0(flags,u,NULL, f1,f2,f3,f4,f5,f6,f7, off1,off2);
		return;
	}

	computeDerivatives(methodDeriv1st,methodDeriv2nd, flags, u,u, g1,g2);

	// debug output
	if(0){ int j=t, k=0; 
	std::cerr <<" 		u  " << u(s-3,j-1,k) <<"  "<< u(s-2,j-1,k) <<"  "<< u(s-1,j-1,k)<<"  "<< u(0,j-1,k) <<"  "<< u(1,j-1,k)   <<'\n';
	std::cerr <<" 		g1 " <<  g1(s-3,j,k) <<"  "<<  g1(s-2,j,k) <<"  "<<  g1(s-1,j,k)<<"  "<<  g1(0,j,k) <<"  "<< g1(1,j,k)   <<'\n';
	std::cerr <<" 		g2 " <<  g2(s-3,j,k) <<"  "<<  g2(s-2,j,k) <<"  "<<  g2(s-1,j,k)<<"  "<<  g2(0,j,k) <<"  "<< g2(1,j,k)    <<'\n'; 
	//std::cerr <<" 		u  " << get(u,s-3,j-1,k) <<"  "<< get(u,s-2,j-1,k) <<"  "<< get(u,s-1,j-1,k)<<"  "<< get(u,0,j-1,k) <<"  "<< get(u,1,j-1,k)   <<'\n';
	//std::cerr <<" 		g1 " <<  get(g1,s-3,j,k) <<"  "<<  get(g1,s-2,j,k) <<"  "<<  get(g1,s-1,j,k)<<"  "<<  get(g1,0,j,k) <<"  "<< get(g1,1,j,k)   <<'\n';
	//std::cerr <<" 		g2 " <<  get(g2,s-3,j,k) <<"  "<<  get(g2,s-2,j,k) <<"  "<<  get(g2,s-1,j,k)<<"  "<<  get(g2,0,j,k) <<"  "<< get(g2,1,j,k)    <<'\n'; 
	}

	FOR_IJK(flags) {
		if(t!=j) continue;		

		Real v = get(u, i,j-1,k );
		v += dt*( -1. * get(u, i,j-1,k) * get(g1, i,j,k) );
		if(methodDeriv2nd==0) { // explicit
			v += dt*( nu * get(g2, i,j,k) );
		} 
		get(u, i,j,k ) = v;
		//get(u, i,j,k ) = get(u, i,j-1,k ); // debug, update off, copy only
	}

	// add afterwards
	if(t0>0 && contForcing) {
		addForcing(flags,u, f1,f2,f3,f4,f5,dt*f6,f7, off1,off2);
	}

	// sanity check, abort when diverging
	FOR_IJK(flags) {
		if( ( fabs(u(i,j,k))>1e6 ) || ( fabs(g1(i,j,k))>1e6 ) || ( fabs(g2(i,j,k))>1e6 ) ) errMsg("Calculation diverged, aborting...");
	}

	// burgers end
}




// u velocity, d density (passive scalar), g1/g2 1st 2nd derivs
// see "burger" above for detailed parameters, new ones:
// offd & fd = density init offset and frequency
PYTHON() void advdiff(FlagGrid& flags, Grid<Real>& u, Grid<Real>& dens, Grid<Real>& g1, Grid<Real>& g2 ,
					int methodDeriv1st=0, int methodDeriv2nd=0, Real nu=0.05,
					Real f1=0., Real f2=1., Real f3=1.,  Real f4=1.,  Real f5=1. , Real f6=1. , Real f7=1.,
					Real off1=0. , Real off2=0. , Real offd=0. , Real fd=1. , bool contForcing=false )
{
	const int t0 = flags.getParent()->mFrame;
	const int s  = flags.getSizeX();
	int       t  = t0; while(t>=s) t-=s;

	const Real dx = 1. / Real(u.getSizeX());
	const Real dt = flags.getParent()->mDt;

	// init sines , adv diff
	if(t0==0) {
		initT0(flags,u,&dens, f1,f2,f3,f4,f5,f6,f7, off1,off2, offd,fd);
		return;
	}

	computeDerivatives(methodDeriv1st, methodDeriv2nd, flags, u, dens, g1,g2);

	// adv-diff , actual time integration
	FOR_IJK(flags) {
		if(t!=j) continue;		

		Real x = get(dens, i,j-1,k );
		//if(i==2) { std::cerr <<"    u " <<j<<" at "<<i <<"  =  "<< v <<'\n'; }

		x += dt*( -1. * get(u, i,j-1,k ) * get(g1, i,j,k) );
		x += dt*( nu * get(g2, i,j,k) );
		get(dens, i,j,k ) = x;
	}

	// adv-diff: vel update
	FOR_IJK(flags) {
		if(t!=j) continue;
		get(u, i,j,k ) = get(u, i,j-1,k );
	}
	if(t0>0 && contForcing) {
		addForcing(flags,u, f1,f2,f3,f4,f5,dt*f6,f7, off1,off2);
	}

	// sanity check, abort when diverging
	FOR_IJK(flags) {
		if( ( fabs(u(i,j,k))>1e6 ) || ( fabs(g1(i,j,k))>1e6 ) || ( fabs(g2(i,j,k))>1e6 ) || ( fabs(dens(i,j,k))>1e6 ) ) 
			errMsg("Calculation diverged, aborting...");
	}

	// adv diff end
}







void setRandomUniform(Vec3& value, RandomStream& mRand, Real strength) {
	value = (Vec3(-0.5) + mRand.getVec3()) * strength;
}

void setRandomUniform(Real& value, RandomStream& mRand, Real strength) {
	value = (mRand.getReal() - 0.5f) * strength;
}

void setRandomNormal(Vec3& value, RandomStream& mRand, Real strength) {
	Real mean = 0;
	Real var = (1.5*strength) * (1.5*strength);
	value = Vec3(mRand.getRandNorm(mean, var), mRand.getRandNorm(mean, var), mRand.getRandNorm(mean, var));
}

void setRandomNormal(Real& value, RandomStream& mRand, Real strength) {
	Real mean = 0;
	Real var = (1.5*strength) * (1.5*strength);
	value = mRand.getRandNorm(mean, var);
}

static int cnt = 0;
PYTHON() void createRandomField(GridBase& noise, Real strength, Shape* excludeShape = NULL, FlagGrid* excludeGrid = NULL, int bWidth = 2, std::string mode = "uniform", int seed = 1)
{
	//KnApplyForceField(flags, vel, force, region, true, isMAC);
	RandomStream mRand(seed + cnt);
	cnt += 10;

	Grid<Real>* noiseReal = NULL;
	MACGrid* noiseMac = NULL;
	if (noise.getType() & GridBase::TypeReal) {
		noiseReal = dynamic_cast< Grid<Real>* >(&noise);
	}
	else if (noise.getType() & GridBase::TypeMAC) {
		noiseMac = dynamic_cast< MACGrid* >(&noise);
	}

	if (mode == "uniform") {
		FOR_IJK_BND(noise, bWidth) {
			if ( (!excludeShape || !excludeShape->isInsideGrid(i, j, k)) &&
				 (!excludeGrid || !excludeGrid->isFluid(i, j, k)) ) {
				if (noiseReal) {
					setRandomUniform((*noiseReal)(i, j, k), mRand, strength);
				}
				else if (noiseMac) {
					setRandomUniform((*noiseMac)(i, j, k), mRand, strength);
				}
			}
		}
	}

	else if (mode == "normal") {
		FOR_IJK_BND(noise, bWidth) {
			if ((!excludeShape || !excludeShape->isInsideGrid(i, j, k)) &&
				(!excludeGrid || !excludeGrid->isFluid(i, j, k)) ) {
				if (noiseReal) {
					setRandomNormal((*noiseReal)(i, j, k), mRand, strength);
				}
				else if (noiseMac) {
					setRandomNormal((*noiseMac)(i, j, k), mRand, strength);
				}
			}
		}
	}

	else {
		throw std::invalid_argument("Unknown mode: use \"uniform\" or \"normal\" instead!");
	}
}

//static int cnt2 = 0;
PYTHON() void applyShapeRandomized(Grid<Real>& grid, Real value, Shape& shape, Real strength, int seed = 1)
{
	RandomStream mRand(seed);
	//cnt2 += 1;

	FOR_IJK(grid) {
		if (shape.isInsideGrid(i, j, k)) {
			grid(i, j, k) = value + mRand.getReal() * strength;
		}
	}
}


PYTHON() void applyNoiseLine(Grid<Real>& grid, Grid<Real>& noise) {
	const int t0 = grid.getParent()->mFrame;
	const int s = grid.getSizeX();
	int       t = t0; while (t >= s) t -= s;
	FOR_IJK(grid) {
		if (t != j) continue;
		grid(i, j, k) += noise(i, j, k);
	}
}

// computes cos( (w+0.1) * x) * e^(-(3.7/r) * x)
// where w is wavyness, r is radius and x is length of translated positon vector
PYTHON() void applyWaveToGrid(Grid<Real>& grid, const Vec3& center, Real radius, Real waviness)
{
	FOR_IJK(grid) {
		Real pos = std::sqrt((i-center[0]) * (i-center[0]) + (j-center[1]) * (j-center[1]) + (k-center[2]) * (k-center[2]));
		grid(i, j, k) += cos( (waviness+0.1) * pos) * std::pow(M_E, -(3.7/radius) * pos);
	}
}


PYTHON()
void saveGridToCSV(const FlagGrid &flags, const std::string fileName, GridBase &gridBase, const int component = 0) {
	ofstream file;
	file.open(fileName);

	Grid<Real>* gridReal = NULL;
	FlagGrid* gridFlags = NULL;
	MACGrid* gridMac = NULL;
	Grid<Vec3>* gridVec3 = NULL;

	if (gridBase.getType() & GridBase::TypeReal) {
		gridReal = dynamic_cast< Grid<Real>* >(&gridBase);
	}
	else if (gridBase.getType() & GridBase::TypeFlags) {
		gridFlags = dynamic_cast< FlagGrid* >(&gridBase);
	}
	else if (gridBase.getType() & GridBase::TypeMAC) {
		gridMac = dynamic_cast< MACGrid* >(&gridBase);
	}
	else if (gridBase.getType() & GridBase::TypeVec3) {
		gridVec3 = dynamic_cast< Grid<Vec3>* >(&gridBase);
	}

	for (int i = 0; i < gridBase.getSizeX(); i++) {
		std::string line = "";
		for (int j = 0; j < gridBase.getSizeY(); j++) {
			IndexInt index = gridBase.index(i, j, 0);
			//if (!(flags[index] & FlagGrid::TypeFluid)) continue;

			if (gridReal) {
				line.append(std::to_string((*gridReal)[index]).append(","));
			}
			else if (gridFlags) {
				if (gridFlags->isFluid(index)) {
					line.append(std::to_string(1).append(","));
				}
				else {
					line.append(std::to_string(0).append(","));
				}
			}
			else if (gridMac) {
				line.append(std::to_string((*gridMac)[index][component]).append(","));
			}
			else if (gridVec3) {
				line.append(std::to_string((*gridVec3)[index][component]).append(","));
			}
		}

		if (!line.empty()) {
			line.pop_back();
			file << line << "\n";
		}
	}

	file.close();
}
/*

PYTHON()
void saveGridToCSV(const FlagGrid &flags, const MACGrid &grid, const int component, const std::string fileName) {
ofstream file;
file.open(fileName);

for (int i = 0; i < grid.getSizeX(); i++) {
std::string line = "";
for (int j = 0; j < grid.getSizeY(); j++) {
IndexInt index = grid.index(i, j, 0);
if (!(flags[index] & FlagGrid::TypeFluid)) continue;

line.ap	pend(std::to_string(grid[index][component]).append(","));
}

if (!line.empty()) {
line.pop_back();
file << line << "\n";
}
}

file.close();
}

PYTHON()
void setGridValue(Grid<Real>& grid, Vec3& pos, Real value) {
	grid(pos.x, pos.y, pos.z) = value;
}

PYTHON()
void setMACValue(MACGrid& grid, Vec3& pos, Vec3& value) {
	grid(pos.x, pos.y, pos.z) = value;
}*/

} //namespace

