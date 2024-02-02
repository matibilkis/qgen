samples_real = all_samples_real[0]
samples_real.shape

discriminator(samples_real.squeeze().unsqueeze(0))

for epoch in tqdm(range(100)):

    all_samples_real, probs_real = give_samples_real(M=M)
    all_samples_real = all_samples_real.split(m)

    for samples_real in all_samples_real:
        probs_fake, samples_qgen = call_qgen(qnn_gen,shots=len(samples_real))

        disc_optimizer.zero_grad()
        disc_on_real = discriminator(samples_real)
        disc_on_fake = discriminator(samples_qgen.unsqueeze(-1))
        cost_discc = cost_discriminator(samples_real,disc_on_fake,disc_on_real,M=M)
        cost_discc.backward()
        disc_optimizer.step()

    ## Evaluate on whole epoch
    probs_fake, samples_qgen = call_qgen(qnn_gen,shots=len(all_samples_real))
    disc_on_real = discriminator(samples_real)
    disc_on_fake = discriminator(samples_qgen.unsqueeze(-1))
    metrics["disc_on_real"].append(torch.mean(disc_on_real))
    metrics["disc_on_fake"].append(torch.mean(disc_on_fake))
    costs["disc"].append(cost_discc)

    for k in range(1):
        gen_optimizer.zero_grad()
        type_qgen = qnn_gen(torch.tensor([]))
        probs_fake, samples_qgen = call_qgen(qnn_gen)
        disc_on_fake = discriminator(samples_qgen.unsqueeze(-1)).detach()

        cost_genn = cost_generator(disc_on_fake, probs_fake)
        cost_genn.backward()
        gen_optimizer.step()
    costs["gen"].append(cost_genn)



#plt.plot(torch.tensor(metrics["disc_on_real"]))
#plt.plot(torch.tensor(metrics["disc_on_fake"]))

plt.plot(torch.tensor(costs["gen"]).detach().numpy())
plt.plot(type_qgen.detach().numpy())
