module globals
        use type_crystalstructure, only: lo_crystalstructure
        use type_forceconstant_secondorder, only: lo_forceconstant_secondorder
        use lo_memtracker, only: lo_mem_helper
        use type_qpointmesh, only: lo_qpoint
        use mpi_wrappers, only: lo_mpi_helper, lo_stop_gracefully
        use lo_phonon_bandstructure_on_path, only: lo_phonon_bandstructure
        use type_phonon_dispersions, only: lo_phonon_dispersions_qpoint
        use konstanter, only: r8

        implicit none

        type(lo_crystalstructure) :: uc
        type(lo_mem_helper) :: mem
        type(lo_forceconstant_secondorder) :: fc
        type(lo_mpi_helper) :: mw
        type(lo_phonon_dispersions_qpoint) :: p
end module

subroutine tdep_initialize()
        use globals
        implicit none
        call mem%init()
        call uc%readfromfile('infile.ucposcar', verbosity=2)
        call fc%readfromfile(uc, 'infile.forceconstant', mem, verbosity=2)
end subroutine

subroutine tdep_compute(qcart, omega, U)
        use globals
        implicit none
        real(r8) qcart(3)
        real(r8) omega(*)
        complex(r8) U(*)
        integer N
        call p%generate(fc, uc, mem, qvec=qcart)
        N = uc%na * 3
        call dcopy(N, p%omega, 1, omega, 1)
        call zcopy(N*N, p%egv, 1, U, 1)
end subroutine

!program tdep_test
!        use globals
!        implicit none
!        real(r8), dimension(3) :: qcart
!        real(r8), dimension(6) :: omega
!        complex(r8), dimension(6,6) :: U
!        call tdep_initialize()
!        qcart(1) = 0.0
!        qcart(2) = 0.0
!        qcart(3) = 0.0
!
!        call tdep_compute(qcart,omega, U)
!
!        PRINT *, omega
!        PRINT *, U
!end program
