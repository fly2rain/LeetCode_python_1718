function val = EuError(R, M, S)

val = 0.5*norm(R - M*S,'fro')^2;