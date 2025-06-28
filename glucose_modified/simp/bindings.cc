#include <pybind11/pybind11.h>

namespace py = pybind11;


PYBIND11_MODULE(glucose_python, m) {
    m.doc() = "python bindings for glucose SAT solver";

}