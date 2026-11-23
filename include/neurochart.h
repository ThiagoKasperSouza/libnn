#ifndef NEUROCHART_H
#define NEUROCHART_H

/*
 * neurochart.h — header público unificado da libnn.
 *
 * Marco 1 (atual): arena de memória, tensores, RNG e álgebra linear.
 * Os demais módulos (autograd, células, topologia, treinadores, zoo e
 * render) serão incluídos aqui conforme seus marcos forem implementados.
 */

#define NC_VERSION_MAJOR 0
#define NC_VERSION_MINOR 1
#define NC_VERSION_PATCH 0

#include "nc_arena.h"
#include "nc_tensor.h"

#endif /* NEUROCHART_H */
