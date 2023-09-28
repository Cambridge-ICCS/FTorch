# compiler
# Note - this should match the compiler that the library was built with
FC = gfortran

# compile flags
FCFLAGS = -O3 -I</path/to/installation>/include/ftorch

# link flags
LDFLAGS = -L</path/to/installation>/lib/ -lftorch

PROGRAM = simplenet_infer_fortran
SRC = simplenet_infer_fortran.f90
OBJECTS = $(SRC:.f90=.o)

all: $(PROGRAM)

$(PROGRAM): $(OBJECTS)
		$(FC) $(FCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.f90
		$(FC) $(FCFLAGS) $(LDFLAGS) -c $<

clean:
		rm -f *.o *.mod

