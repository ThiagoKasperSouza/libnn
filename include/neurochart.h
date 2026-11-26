#ifndef NEUROCHART_H
#define NEUROCHART_H

/*
 * neurochart.h — header público unificado da libnn.
 *
 * Marcos 1 e 2: arena de memória, tensores, RNG, álgebra linear, fita de
 * autograd, células feedforward, topologias, Adam e construtores zoo.
 * Os demais módulos (operações espaciais otimizadas, tempo, energia e
 * render) serão incluídos aqui conforme seus marcos forem implementados.
 */

#define NC_VERSION_MAJOR 0
#define NC_VERSION_MINOR 1
#define NC_VERSION_PATCH 0

#include "nc_arena.h"
#include "nc_tensor.h"
#include "nc_autograd.h"
#include "nc_cells.h"
#include "nc_topology.h"
#include "nc_trainers.h"
#include "nc_sequence.h"
#include "nc_zoo.h"
#include "nc_render.h"

#endif /* NEUROCHART_H */
