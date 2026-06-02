// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/parse_cmd_options.h>
#include <prismspf/core/problem.h>
#include <prismspf/core/solve_block.h>

using namespace prismspf;

int
main(int argc, char *argv[])
{
  // Initialize MPI
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv,
                                                    dealii::numbers::invalid_unsigned_int);

  // Restrict deal.II console printing
  dealii::deallog.depth_console(0);

  // Parse the command line options (if there are any) to get the name of the
  // input file
  ParseCMDOptions cli_options(argc, argv);

  constexpr unsigned int dim    = 2;
  constexpr unsigned int degree = 1;

  std::vector<FieldAttributes> fields = {FieldAttributes("n"),
                                         FieldAttributes("x1"),
                                         FieldAttributes("rxn_mu"), 
                                         FieldAttributes("rxn"),
                                         FieldAttributes("x1_stage")
                                         };

  SolveBlock main_fields(0, Explicit, Initialized, {0, 1});
  main_fields.dependencies_rhs = make_dependency_set(
      fields, {"old_1(n)", "old_1(x1_stage)", "old_1(rxn)", "old_1(x1)"});
      
  SolveBlock potential(1, Explicit, Uninitialized, {2});
  potential.dependencies_rhs = make_dependency_set(fields, {"n", "grad(n)", "x1"});

  SolveBlock rxn(2, Explicit, Uninitialized, {3});
  rxn.dependencies_rhs = make_dependency_set(fields, {"n", "grad(n)", "rxn_mu"});

  SolveBlock x1_stage(3, Linear, Uninitialized, {4});
  x1_stage.dependencies_rhs = make_dependency_set(fields, {"n", "grad(n)", "x1", "grad(x1)", "rxn"});
  x1_stage.dependencies_lhs = make_dependency_set(fields, {"lhs(x1_stage)", "grad(lhs(x1_stage))", "n", "grad(n)"});

  std::vector<SolveBlock> solve_blocks({main_fields, potential, rxn, x1_stage});

  UserInputParameters<dim>       user_inputs(cli_options.get_parameters_filename());
  PhaseFieldTools<dim>           pf_tools;
  CustomPDE<dim, degree, double> pde_operator(user_inputs, pf_tools);
  Problem<dim, degree, double>   problem(fields, solve_blocks, user_inputs, pf_tools, pde_operator);
  problem.solve();

  return 0;
}
