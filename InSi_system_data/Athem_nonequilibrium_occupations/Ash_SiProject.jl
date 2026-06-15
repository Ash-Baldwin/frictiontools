using LightMatter, OrdinaryDiffEq, RefractiveIndex, Unitful

SiRI = RefractiveMaterial("https://refractiveindex.info/?shelf=main&book=Si&page=Franta-250K")

Si = (bulk_DOS = "Si_DOS.dat" , bulk_geometry = "geometry_bulk.in", name="Si")

hv = 1.0
Dim =  build_Dimension([0.0])
mat = Si #This is because I often do for mat in [Au, Ag] so couldn't be bothered to change everything to Si.x rather than mat.x


#Laser Settings
λ = Constants.ħ * 2π * Constants.c / hv  #nm
mat_n = SiRI(λ * u"nm")
mat_κ = extinction(SiRI, λ*1e-3)
mat_R = ((mat_n - 1)^2 + mat_κ^2) / ((mat_n + 1)^2 + mat_κ^2)
mat_ϵ = λ ./ (4*pi*mat_κ)
ϕ = ustrip(uconvert(u"eV/nm^2", 1u"mJ/cm^2"))
Las = build_Laser(envelope=:Gaussian, FWHM=50.0, ϕ=ϕ, hv=hv, Transport=:optical, ϵ=mat_ϵ, R = mat_R)

#Electronic and Geometric Structure
egrid = collect(range(-2*hv, 2*hv, length=7500))
output = "Ash_Calc.hdf5"
Struc = build_Structure(las = Las, dimension=Dim, DOS_file = mat.bulk_DOS, DOS_geometry = mat.bulk_geometry, 
                        Elemental_System=1, egrid=egrid, calculate_bandstructure=false)

#What you want to propagate
Neq = build_AthermalElectrons(Enabled=true, structure=Struc, ExcitationMatrixElements=:unity)    

#Running Simulation with initial conditions
Sim = build_Simulation(athermalelectrons = Neq, laser=Las, structure=Struc)
initialtemps=Dict("Tel"=>300.0,"Tph"=>300.0)
tspan=(-200.0, 200.0)
sol = run_simulation(Sim, initialtemps, tspan, dtmin=1e-3, reltol=1e-8, abstol=1e-8, saveat=1.0, alg=Vern7())#alg=Rodas5P()
post_production(sol,output,initialtemps,Sim)


fneq = sol[end].fneq[1,:]
egrid_new = Sim.structure.egrid #Integration requires that egrid actual is not necessarily the same as the one you defined e.g. has to be certain length
DOS = Sim.structure.DOS(egrid_new)

f = fneq.*DOS
f_uc = f./10^3 .* 5.431^3 #Distribution per bulk unit cell of Si

dE = egrid_new[2] - egrid_new[1]
zero_idx = findfirst(x -> x > 0, egrid_new)
n_exc = cumsum(f_uc[zero_idx:end]) *dE
println("Number of excited electrons per primitive unit cell of Si: ", n_exc[end])

using Plots
plot(egrid_new, f_uc) #./10^3 so that Å⁻³ rather than nm⁻³ (units in this calculation are nm including the DOS)