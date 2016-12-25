// rt: a minimalistic ray tracer
// g++ -O3 -fopenmp rt.cpp -o rt
// remove "-fopenmp" for g++ version < 4.2

//Mcgill ID 260741007
//Name: Aakash Nandi
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cmath>
#include <vector>

//No. of samples to be taken in uniform solid angle sampling
int samples=10000;

//No of bounces that are allowed to take place
int bounces=2;

//Structure definition for a vector
struct Vec
{

	//stores 3 values corresponding to x,y,z;
	double x, y, z;

	//Default contructor
	Vec(double x_= 0, double y_= 0, double z_= 0)
	{
		x=x_;
		y=y_;
		z=z_;
	}

	//Overloading operators to perform vector math
	//Addition
	Vec operator+(const Vec &b) const
	{
		return Vec(x + b.x, y + b.y, z + b.z);
	}

	//Subtraction
	Vec operator-(const Vec &b) const
	{
		return Vec(x - b.x, y - b.y, z - b.z);
	}

	//Multiplication with a double "b"
	Vec operator*(double b) const
	{
		return Vec(x * b, y * b, z * b);
	}

	//Cross product with vector "b" returns vector
	Vec operator%(Vec&b)
	{
		return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
	}

	//Dot product with vector "b" returns double value
	double dot(const Vec &b) const
	{
		return x * b.x + y * b.y + z * b.z;
	}

	//element wise multiplication with vector "b"
	Vec mult(const Vec &b) const
	{
		return Vec(x * b.x, y * b.y, z * b.z);
	}

	//division by a scalar value
	Vec operator/(const int b)
	{
		return Vec(x/b,y/b,z/b);
	}

	//return a unit vector for a given vector
	Vec& norm()
	{
		return *this = *this * (1.0 / sqrt(x * x + y * y + z * z));
	}
};

//Location of the camera
Vec cam(50, 50, 275.0);

//Structure definition for a Ray
struct Ray
{
	//Ray defined by origin point and destination direction defined by unit vector. (Point's coordinate defined by vector)
	Vec o, d;
	//Constructor for Ray that calls copy constructor for vectors "o" and "d"
	Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};


//Structure definition for a Sphere
struct Sphere
{
	//Radius of sphere
	double r;

    //Determines whether the object is diffused or not
	bool diff;

    //If the object is not diffused , this value determines its glossiness
	double phongpower;

	//vectors for p : center e : power that it emits and c : vector of color (RGB) where (z : blue, y : green x : red)
	Vec p, e, c;

	//Constructor for sphere calling copy constructor for its attributes.
	Sphere(double r_, Vec p_, Vec e_, Vec c_, bool type,double phgpwr=0): r(r_), p(p_), e(e_), c(c_) {diff=type;phongpower=phgpwr;}

	//function to check for intersection
	//returns value which when multiplied with ray.d gives point of intersection
	//return 0 if intersection does not occur
	double intersect(const Ray &ray) const
	{
		//vector joining center and origin of ray
		Vec op = p - ray.o;
		double t, eps = 1e-4;

		//magnitude of projection of op on ray.
		double b = op.dot(ray.d);

		//b*b is square of magnitude of projection of op
		//op.dot(op) is square of magnitude of op
		//op.dot(op)-(b*b) give the square of the (distance b/w center and midpoint of intersecting chord) say (p_m)
		//det is difference between square of radius and square of distance "p_m"
		double det = b * b - op.dot(op) + r * r;

		//det turns out to be negative if p_m>r ie no intersection takes place
		//hence return 0
		if (det < 0) return 0;
		//else compute the distance b/w midpoint of chord and point of intersection
		else det = sqrt(det);

		//check whether ray is outside the sphere or on the sphere  and return k to be multiplied to ray for point of interesection.
		//if on the sphere then return second  point of intersection
		return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
	}
};


//Declaration of a array of spheres
Sphere spheres[] =
{
	//flat surface made by large spheres of radius 1e5

	//right side wall  coloured (greenish blue)
	Sphere(1e5,  Vec(1e5 + 99, 40.8, 81.6), Vec(0,0,0), Vec(0.25, 0.25, 0.75),true),

	//left side wall coloured (pink)
	Sphere(1e5,  Vec(-1e5+1 , 40.8, 81.6), Vec(0, 0, 0), Vec(0.75, 0.25,0.25),true),

	//front wall white with intensity 0.75
	Sphere(1e5,  Vec(50, 40.8, -1e5), Vec(0, 0, 0), Vec(0.75, 0.75, 0.75),true),

	//top wall white with intensity 0.75
	Sphere(1e5,  Vec(50, 1e5+81.6, 81.6), Vec(0, 0, 0), Vec(0.75, 0.75, 0.75),true),

	//bottom wall white with intensity 0.75
	Sphere(1e5,  Vec(50, -1e5 , 81.6), Vec(0, 0, 0), Vec(0.75,0.75, 0.75),true),

	//rear most sphere
	Sphere(16.5,  Vec(27, 16.5, 47), Vec(0, 0, 0), Vec(0.999, 0.999, 0.999),true),

	//middle sphere
	Sphere(16.5,  Vec(73, 16.5, 78), Vec(0, 0, 0), Vec(0.999, 0.999, 0.999),false,10000),

	//front most sphere turned into a light source with radiance 10.
	Sphere(10, Vec(50, 68.6 - .27, 81.6), Vec(10, 10, 10), Vec(1, 1, 1),true)
};

//Declaration of pointer to the source of light
Sphere *src;

//0 if <0 , 1 if >1,else value
inline double clamp(double x)
{
	return x < 0 ? 0 : x > 1 ? 1 : x;
}

//some mathematical normalisation
inline int toDisplayValue(double x)
{
	return int( pow( clamp(x), 1.0/2.2 ) * 255 + .5);
	//.5 added to round off (decimal less than 0.5 -> rounded to lower integer and likewise.)

}

//Find out which sphere is the first to intersect with the ray and also return multiplier for the ray
//also returns a boolean check for intersection.
inline bool intersect(const Ray &r, double &t, int &id)
{
	//number of spheres in the array
	double n = sizeof(spheres) / sizeof(Sphere);
	double d, inf = t = 1e20;
	id=-1;

	//for each sphere find the intersection multiplier for ray
	//store the id of sphere that intersects first.
	for(int i = int(n); i--; )
	{
		if( (d = spheres[i].intersect(r)) && d < t)
		{
			t = d;
			id = i;
		}
	}

	//boolean check for intersection
	return t < inf;
}



/*
checks whether the light ray between the object an source
is unhindered by any other object.
*/
int visible(Vec x)
{
	Ray x2src(x,(src->p-x).norm());
	double t;
	int id;
	//Source connecting x to source
	//x2src.o=(x);
	//x2src.d=(src->p-x).norm;
	intersect(x2src,t,id);
	if(src==&spheres[id])
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/*to compute tilted sampling ray
takes in vector a which is sample ray wrt (0,0,1)
and vector n which is the normal at the point of intersection.
 */
Vec tilt(Vec a,Vec n)
{
Vec cro(0,0,0),xtran(0,0,0),ytran(0,0,0),ztran(0,0,0),stdn(0,0,1);
double cos;
cro=stdn%n;
cos=stdn.dot(n);

//first row of rotation matrix
xtran.x=1.0-((cro.z*cro.z)+(cro.y*cro.y))/(1.0+cos);
xtran.y=((cro.x*cro.y)/(1.0+cos))-cro.z;
xtran.z=((cro.x*cro.z)/(1.0+cos))+cro.y;

//second row of rotation matrix
ytran.x=((cro.x*cro.y)/(1.0+cos))+cro.z;
ytran.y=1.0-(((cro.x*cro.x)+(cro.z*cro.z))/(1.0+cos));
ytran.z=((cro.y*cro.z)/(1.0+cos))-cro.x;

//third row of rotation matrix
ztran.x=((cro.x*cro.z)/(1.0+cos))-cro.y;
ztran.y=((cro.y*cro.z)/(1.0+cos))+cro.x;
ztran.z=1.0-(((cro.x*cro.x)+(cro.y*cro.y))/(1.0+cos));

return Vec(xtran.dot(a),ytran.dot(a),ztran.dot(a));
}


/*
To compute the reflected unit vector at a given point
considering the camera to be at (50,50,275)
*/

Vec reflect(Vec x,Vec n)
{
Vec d,r;
d=x*(-1);
r=n*(2*d.dot(n))-d;
return r.norm();
}

/*
Random unit vector generator
for uniform spherical sampling
*/
Vec generatevec_diff(Vec n)
{

	double theta,phi,e1,e2,x=0,y=0,z=0;
	e1=((double)rand()/(double)(RAND_MAX));
	e2=((double)rand()/(double)(RAND_MAX));

	//Phi is 2*pi*e1
	phi=2*3.14*e1;

	//Theta is argcos e2
	theta=acos(e2);

	//printf("thetan : %f phin : %f theta : %f phi : %f\n",thetan,phin,theta,phi);
	x=sin(theta)*cos(phi);
	y=sin(theta)*sin(phi);
	z=cos(theta);


	Vec hld(x,y,z);
	hld=tilt(hld,n);
	return hld.norm();
}

/*
Random unit vector generator
for phong lobe sampling
*/
Vec generatevec(Vec n,double phong)
{

	double theta,phi,e1,e2,x=0,y=0,z=0;
	e1=((double)rand()/(double)(RAND_MAX));
	e2=((double)rand()/(double)(RAND_MAX));

	//Phi is 2*pi*e1
	phi=2*3.14*e1;

	//Theta is argcos e2
	theta=acos(pow(1-e2,double(1.0/double(phong+1))));

	//printf("thetan : %f phin : %f theta : %f phi : %f\n",thetan,phin,theta,phi);
	x=sin(theta)*cos(phi);
	y=sin(theta)*sin(phi);
	z=cos(theta);


	Vec hld(x,y,z);
	hld=tilt(hld,n);
	return hld.norm();
}

/*
For a given ray and starting point, bounce that ray as per the number the bounces
and return the intensity
*/
Vec throwray(Ray initray,int bounces_left)
    {
    Vec current_intensity(0,0,0),sample_new,pt_intersect,n_intersect;
    double t,cos_v;
    int id;
    if (bounces_left==0)
        {
        intersect(initray,t,id);
        if(&spheres[id]!=src){return current_intensity;}
        else{return src->e;}
        }
    else
        {
        intersect(initray,t,id);
        if(&spheres[id]==src){return src->e;}
        else
            {
            pt_intersect=initray.o+initray.d*t;
            n_intersect=(pt_intersect-(spheres[id].p)).norm();
            if(spheres[id].diff)
            {
            sample_new=generatevec_diff(n_intersect);
            }
            else
            {
            sample_new=generatevec(reflect(initray.d,n_intersect),spheres[id].phongpower);
            }

            current_intensity=throwray(Ray(pt_intersect,sample_new),bounces_left-1);
            cos_v=sample_new.dot(n_intersect);
            if(spheres[id].diff)
            {
            return (spheres[id].c).mult(current_intensity*cos_v*double(2*3.14)/double(3.14));
            }
            else
            {
            return (spheres[id].c).mult(current_intensity*cos_v);
            }
            }
        }
    }



/*
For a given point in the scene
it perform uniform spherical sampling
and returns the intensity
*/
Vec samplelight(Vec x,Sphere obj,int id)
{
	double cos;
	int i;
	Ray sample_ray(0,Vec());
	Vec intensity_bank,intensity_obtained;

	//vector joining point of intersection on source
	//and the given point x
	Vec src_pnt;


	//find the unit vector normal to the surface at the point of intersection
	Vec n = (x - obj.p).norm();

	//perform sampling and store intensity
	//in intensity_bank
	for(i=0; i<samples; i++)
	{

		//sample ray
		//if diffused, perform hemispherical sampling
		//else phong lobe sampling
		if(spheres[id].diff)
		{
		sample_ray=Ray(x,generatevec_diff(n));
		}
		else
		{
		sample_ray=Ray(x,generatevec(reflect((x-cam).norm(),n),spheres[id].phongpower));
        }

		//replace the intersect function with traceback
		//function to throw the ray(number of bounces)
		intensity_obtained=throwray(sample_ray,bounces);

        //find the cosine between the normal and the vector joining the intersection to the center
        cos=(sample_ray.d).dot(n);

        //intensity contribution due to a point on light source  (divided by pdf and multiplied by brdf)
        if(spheres[id].diff)
        {
        intensity_bank=intensity_bank+((intensity_obtained)*cos)*double(2*3.14)/double(3.14);
        }
        else
        {
         intensity_bank=intensity_bank+((intensity_obtained)*cos);
        }

	}

	//normalize the intensity sum in intensity_bank by samples
	intensity_bank=intensity_bank/samples;
	return intensity_bank;
}

/*
The following function below is resposible for colouring the whole scenario.
First the point of intersection is found out (represented using vector x).
Then a unit vector joining the point of intersection and center is computed (represented by vector n).
This vector is added to the vector c of the intersecting sphere.
For example.
Q) why is the left side of sphere cyan?
A) A vector joining the left center point would be (-1,0,0)
    and the sphere's colour vector ie (R:1,G:1,B:1) when added
    would yield (R:0,G:1,B:1) is cyan.

Q) why is the center at front white?
A) A vector joining the front center would be (0,0,1)
    and the sphere's colour vector (R:1,G:1,B:1) when added
    would yield (R:1,G:1,B:2) which when subjected to clamp
    function would yield (R:1,G:1,B:1) ie white.
*/
Vec shade(const Ray &r)
{
	double t;
	int id = 0;
	Vec rec_intensity;
	//checks for the intersection of ray with any of the spheres
	//sets id and value of t for first intersection
	if (!intersect(r, t, id))
		return Vec();

	//select the sphere which hits the ray first
	const Sphere &obj = spheres[id];

	//corner case to render the source
	if(&obj==src)
	{
		return obj.c;
	}

	//find the point of intersection on the sphere
	Vec x = r.o + r.d * t;

	//recorded intensities after performing sampling
	rec_intensity=samplelight(x,obj,id);

	//colour vector multiplied with the received intensity and then normalised
	return (obj.c).mult(rec_intensity);
}


int main(int argc, char *argv[])
{

	//width and height of the image
	int w = 512, h = 384;
	src=&spheres[7];

	//position and direction of the camera
	Ray camera( Vec(50, 50, 275.0), Vec(0, -0.05, -1).norm() );

	//please refer to explanation_1 first.
	/*
	As per the given hint it seems that the height is substending
	an angle of 30 degrees on the camera .
	so if "h" can substend 30 degree then what angle will the
	x-displacement substend ?
	A) lets say the adjacent side of h is A such that
	    h/A=tan(30) then what will x/A be.
	    A=h/tan(30).

	    substituting A we get:
	    x*tan(30)/h.
	    this means that if we move ahead by 1 on Z - axis the
	    the movement on X-axis will be (x*tan(30)/h)

	    notice that cx is being multiplied to a fraction (x-displacement from camera/w)
	    so in order to get just x-displacement we have w pre-multiplied
	    thus (w*tan(30)/h).
	*/
	Vec cx = Vec( w * 0.57735 / h, 0., 0.);
	// hint : tan( 30 / 180.0 * M_PI ) == 0.57735

	/*
	what comes to be multiplied is already
	fraction (y-displacement from camera/h )
	so we donot multiply w in this case.
	The previously explained math applies here too.
	*/
	Vec cy = (cx % camera.d).norm() * 0.57735;

	//pixelColors holds the value of color of each pixel
	Vec pixelValue, *pixelColors = new Vec[w * h];

	#pragma omp parallel for schedule(dynamic, 1) private(pixelValue)
	for(int y = 0; y < h; y++)
	{

		fprintf(stderr,"\r%5.2f%%",100.*y/(h-1));
		for(int x = 0; x < w; x++ )
		{
			//1D index for 2D indices
			//image has 0,0 at the bottom-left
			//idx starts from top-left
			int idx = (h - y - 1) * w + x;

			pixelValue = Vec();

			//compute the direction of ray joining the pixel and camera
			/*
			explanation_1:

			 (double(x)/w- .5) : what fraction of w is the x-displacement between the camera
			 and the given pixel,considering camera is placed at the center ie (w/2,w/2).
			 Its basically (1/w)*(x-(w/2)).

			 (double(y)/h- .5) : what fraction of h is the y-displacement between the camera
			 and the given pixel,considering camera is placed at the center ie (w/2,w/2).
			 Its basically (1/h)*(y-(h/2)).

			*/
			Vec cameraRayDir = cx * ( double(x)/w - .5) + cy * ( double(y)/h - .5) + camera.d;

			/*we need to move the camera ray to the bulb ray*/

			//make the ray intersect with the scenario to return the colour of the pixel
			pixelValue = shade( Ray(camera.o, cameraRayDir.norm()) );

			//normalise the colour vector and store in i=the pixelColour array.
			pixelColors[idx] = Vec(clamp(pixelValue.x), clamp(pixelValue.y), clamp(pixelValue.z));
		}
	}

	fprintf(stderr,"\n");

	// hint: Google the PPM image format
	FILE *f = fopen("image.ppm", "w");
	//P3 stands for number in ascii format
	//w h specify dimension of the image
	//255 specifies the maximum value
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
	for (int p = 0; p < w * h; p++)
	{
		//write space separated RGB values of a pixel
		fprintf(f,"%d %d %d ", toDisplayValue(pixelColors[p].x), toDisplayValue(pixelColors[p].y), toDisplayValue(pixelColors[p].z));
	}
	fclose(f);

	//empty space by deleting pixelColours
	delete pixelColors;

	return 0;
}
